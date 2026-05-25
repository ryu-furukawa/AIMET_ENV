import os
import json
import pandas as pd

base_dir = "/root/AIMET_ENV/tools/ruri/convert/script/jmteb_result/0521"

results = {}

for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        json_file = os.path.join(subdir_path, "summary_768d.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            results[subdir] = {task: scores["ndcg@10"] for task, scores in data["Retrieval"].items()}

df = pd.DataFrame(results).T

import pandas as pd

pd.set_option("display.float_format", "{:.10f}".format)

print(df)