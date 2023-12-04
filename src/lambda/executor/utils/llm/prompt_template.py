
CLAUDE2_RAG_PROMPT_TEMPLTE = """
Here is a query:

<query>
{query}
</query>

Here are some results relevant to the query:

<search_results>
{context}
</search results>

Once again, the user's query is:

<query>
{query}
</query>

Provide a response between <result> tags.
"""

CLAUDE2_RAG_CONTEXT_TEMPLATE="""
<item index="{index}">
{content}
</item>
"""


def claude2_rag_template_render(query:str,contexts:list):
    """use claude2 offical rag prompte template

    Args:
        query (str): _description_
        contexts (list): _description_
    """
    assert isinstance(contexts,list), contexts
    context_xmls = []
    for i,context in enumerate(contexts):
        context_xml = CLAUDE2_RAG_CONTEXT_TEMPLATE.format(
            index = i+1,
            content = context
        )
        context_xmls.append(context_xml)
    
    context = "\n".join(context_xmls)
    prompt = CLAUDE2_RAG_PROMPT_TEMPLTE.format(query=query,context=context)
    return prompt






