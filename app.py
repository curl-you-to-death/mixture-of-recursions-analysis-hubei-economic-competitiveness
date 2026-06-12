from flask import Flask, render_template
from mor import get_mor_prediction
from model_compare import get_compare_result
from attention_layers import get_attention_layers_result
from num_of_indicators import get_dimension_result
from ratio_input import get_ratio_result

app = Flask(__name__)

REGIONS = [
    "湖北", "武汉", "黄石", "十堰", "宜昌", "襄阳",
    "鄂州", "荆门", "孝感", "荆州", "黄冈", "咸宁",
    "随州", "恩施", "仙桃", "潜江", "天门", "神农架"
]

@app.route('/')
def index():
    # MoR 单轮训练
    mor_scores = get_mor_prediction()

    # 3种模型各 20轮×300次
    compare_data = get_compare_result()

    # 注意力层数 1/2/3各 20轮×300次
    attn_data = get_attention_layers_result()

    # 6个维度各 20轮×300次
    dim_data = get_dimension_result()

    # 调整输入比例单轮训练
    ratio_data = get_ratio_result()

    # 传给前端
    return render_template(
        "index.html",
        regions=REGIONS,
        mor_scores=mor_scores,
        compare=compare_data,
        attn_data=attn_data,
        dim_data=dim_data,
        ratio_data=ratio_data
    )

if __name__ == '__main__':
    app.run(debug=True)