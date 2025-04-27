# explain.py
import os, openai, torch
from torch_geometric.nn import GNNExplainer
from data_utils import load_sources   # 如果你需要把 tweet_id -> 文本 映射回来

# ---- (1) 解释器：拿关键子图 ----------------------------------------------------
def explain_node(model, data, node_id, topk=10):
    """
    返回与 node_id 最相关的 tweet_id 列表（GNNExplainer 得分最高的 top‑k 边）
    """
    explainer = GNNExplainer(model, epochs=500, log=False)
    node_feat_mask, edge_mask = explainer.explain_node(
        node_id, data.x_dict, data.edge_index_dict)

    # edge_mask 是所有边的权重，[num_edges]；取最大的 topk
    top_edges = edge_mask.topk(topk).indices.tolist()

    # edge_index 是 2 x E；取这些边的起点 tweet_id
    rev_map = {v: k for k, v in data['tweet'].mapping.items()}  # idx -> tweet_id
    important_ids = {
        rev_map[data['tweet', 'TD', 'tweet'].edge_index[0, i].item()]
        for i in top_edges
    }
    return list(important_ids)

# ---- (2) GPT4o 调用：使用 LiteLLM 代理 ----------------------------------------
def gpt_justify(tweet_texts, role_msg="Explain why the model predicts this label"):
    """
    输入： tweet_texts (list[str])
    输出： GPT 生成的自然语言解释
    """
    ### 代理所需的两行 -----------------------------------
    client = openai.OpenAI(
        api_key=os.environ["LITELLM_API_KEY"],          # 你的 key 放到 env
        base_url="https://cmu.litellm.ai"               # 代理地址
    )
    ### --------------------------------------------------

    prompt = (
        "Here is a set of tweets related in the propagation graph:\n\n" +
        "\n".join(f"- {t}" for t in tweet_texts[:15]) +   # 最多传 15 条，避免超长
        f"\n\n{role_msg}. Respond in English."
    )

    rsp = client.chat.completions.create(
        model="gpt-4o-mini",                             # 或 gpt-4o
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return rsp.choices[0].message.content.strip()

# ========== 调用示例 ============================================================
if __name__ == "__main__":
    #1. 加载 hetero_graph.pt, model.pt 等（略）
    model.eval(); node_id = 任意测试节点 …

    #2. 解释：
    important_ids = explain_node(model, data, node_id, topk=10)

    # 3. 把 tweet_id -> 文本（你有 source / 其它文本映射）
    texts = [id2text[t] for t in important_ids]
    print(gpt_justify(texts))
    pass
