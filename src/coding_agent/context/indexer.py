from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import uuid

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import ollama

from coding_agent.config import config


@dataclass
class CodeSymbol:
    file_path: str
    symbol_name: str
    symbol_type: str  # function, class, method, import
    start_line: int
    end_line: int
    code_snippet: str
    docstring: str | None = None


class ASTCodeIndexer:
    """Extracts Python AST symbols using Tree-sitter and indexes into Qdrant."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        qdrant_url: str | None = None,
        collection_name: str | None = None,
    ):
        self.workspace_root = (workspace_root or config.workspace_root).resolve()
        self.qdrant_url = qdrant_url or config.qdrant_url
        self.collection_name = collection_name or config.qdrant_collection
        self.embedding_model = config.embedding_model

        # Initialize Tree-sitter parser
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

        # Initialize Qdrant client
        self.qdrant = QdrantClient(url=self.qdrant_url)
        self.ollama_client = ollama.Client(host=config.ollama_base_url)

    def extract_symbols(self, file_path: Path) -> list[CodeSymbol]:
        """Extract functions, classes, and imports from a Python source file."""
        try:
            source_bytes = file_path.read_bytes()
        except Exception:
            return []

        tree = self.parser.parse(source_bytes)
        source_code = source_bytes.decode("utf-8", errors="replace")
        lines = source_code.splitlines()

        rel_path = str(file_path.relative_to(self.workspace_root))
        symbols: list[CodeSymbol] = []

        def get_text(node: Node) -> str:
            return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        def traverse(node: Node, parent_class: str | None = None):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                func_name = get_text(name_node) if name_node else "anonymous"
                full_name = f"{parent_class}.{func_name}" if parent_class else func_name
                symbol_type = "method" if parent_class else "function"

                snippet = "\n".join(lines[node.start_point[0] : node.end_point[0] + 1])
                symbols.append(
                    CodeSymbol(
                        file_path=rel_path,
                        symbol_name=full_name,
                        symbol_type=symbol_type,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        code_snippet=snippet,
                    )
                )

                # Traverse children for nested functions or statements
                for child in node.children:
                    traverse(child, parent_class=parent_class)

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                class_name = get_text(name_node) if name_node else "anonymous"
                snippet = "\n".join(lines[node.start_point[0] : min(node.start_point[0] + 15, node.end_point[0] + 1)])
                symbols.append(
                    CodeSymbol(
                        file_path=rel_path,
                        symbol_name=class_name,
                        symbol_type="class",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        code_snippet=snippet,
                    )
                )

                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        traverse(child, parent_class=class_name)

            elif node.type in ("import_statement", "import_from_statement"):
                snippet = get_text(node)
                symbols.append(
                    CodeSymbol(
                        file_path=rel_path,
                        symbol_name=snippet.strip(),
                        symbol_type="import",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        code_snippet=snippet,
                    )
                )

            else:
                for child in node.children:
                    traverse(child, parent_class=parent_class)

        traverse(tree.root_node)
        return symbols

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding vector using Ollama or fallback to 768-dim zeros."""
        try:
            res = self.ollama_client.embeddings(
                model=self.embedding_model,
                prompt=text,
            )
            return res["embedding"]
        except Exception:
            return [0.0] * 768

    def ensure_collection(self, vector_dim: int = 768) -> None:
        """Ensure the target Qdrant collection exists."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

    def index_workspace(self) -> int:
        """Scan workspace .py files and index symbols into Qdrant."""
        self.ensure_collection()
        points: list[PointStruct] = []
        total_symbols = 0

        for py_file in self.workspace_root.rglob("*.py"):
            # skip ignored paths
            parts = py_file.relative_to(self.workspace_root).parts
            if any(p.startswith(".") or p in ("node_modules", "dist", "build", "venv", "__pycache__") for p in parts):
                continue

            symbols = self.extract_symbols(py_file)
            for sym in symbols:
                text_to_embed = f"{sym.symbol_type} {sym.symbol_name} in {sym.file_path}\n{sym.code_snippet}"
                vector = self.get_embedding(text_to_embed)
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sym.file_path}:{sym.symbol_name}:{sym.start_line}"))

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=asdict(sym),
                    )
                )
                total_symbols += 1

        if points:
            # Batch upsert
            batch_size = 50
            for i in range(0, len(points), batch_size):
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points[i : i + batch_size],
                )

        return total_symbols

    def search_symbols(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search code symbols matching query semantically."""
        try:
            query_vector = self.get_embedding(query)
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "score": hit.score,
                    "symbol_name": hit.payload.get("symbol_name"),
                    "symbol_type": hit.payload.get("symbol_type"),
                    "file_path": hit.payload.get("file_path"),
                    "start_line": hit.payload.get("start_line"),
                    "end_line": hit.payload.get("end_line"),
                    "code_snippet": hit.payload.get("code_snippet"),
                }
                for hit in response.points
            ]
        except Exception as e:
            return [{"error": f"Search failed: {e}"}]
