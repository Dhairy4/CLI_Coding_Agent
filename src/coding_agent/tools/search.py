from langchain_core.tools import tool
from coding_agent.context.indexer import ASTCodeIndexer


@tool
def search_symbols(query: str, limit: int = 5) -> str:
    """Semantically search codebase functions, classes, and methods via AST vector index.

    Args:
        query: Description or name of the symbol/functionality you are looking for.
        limit: Max number of symbol matches to return.
    """
    try:
        indexer = ASTCodeIndexer()
        results = indexer.search_symbols(query=query, limit=limit)
        if not results:
            return f"No symbols found for query '{query}'."

        formatted = [f"Found {len(results)} relevant symbols:"]
        for r in results:
            if "error" in r:
                return f"Error: {r['error']}"
            formatted.append(
                f"- [{r.get('symbol_type')}] `{r.get('symbol_name')}` in {r.get('file_path')}:{r.get('start_line')}-{r.get('end_line')}"
            )
            snippet = r.get("code_snippet", "")
            if snippet:
                preview = "\n".join(snippet.splitlines()[:5])
                formatted.append(f"  Preview:\n```\n{preview}\n```")

        return "\n".join(formatted)
    except Exception as e:
        return f"Error searching symbols: {e}"
