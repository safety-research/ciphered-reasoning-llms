import os
import psycopg2
import json
import hashlib
import ray


def compute_experiment_hash(config):
    sha1 = hashlib.sha1()
    sha1.update(json.dumps(config["experiment"], sort_keys=True).encode("utf-8"))
    return sha1.hexdigest()


@ray.remote(num_cpus=1)
def save_experiment_meta(config):
    def compute_experiment_hash(config):
        sha1 = hashlib.sha1()
        sha1.update(json.dumps(config["experiment"], sort_keys=True).encode("utf-8"))
        return sha1.hexdigest()

    experiment_hash = compute_experiment_hash(config)

    with open(f"output/{experiment_hash}/payload.json", "r") as f:
        d_payload = json.load(f)

    conn_string = os.environ["SUPABASE_CONNECTION_URL"]

    print(f"Connecting to database\n	-> {conn_string}")

    conn = psycopg2.connect(conn_string)
    conn.autocommit = True

    table_name = config["experiment"]["table_name"]

    d_payload = {**d_payload, **config["experiment"]}

    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO public.{table_name} (data, experiment_hash) VALUES (%s, %s) ON CONFLICT (experiment_hash) DO UPDATE SET data = EXCLUDED.data, experiment_hash = EXCLUDED.experiment_hash, created_at = EXCLUDED.created_at",
        (json.dumps(d_payload), experiment_hash),
    )


@ray.remote(num_cpus=1)
def init_experiment_meta_dict(config):
    def compute_experiment_hash(config):
        sha1 = hashlib.sha1()
        sha1.update(json.dumps(config["experiment"], sort_keys=True).encode("utf-8"))
        return sha1.hexdigest()

    experiment_hash = compute_experiment_hash(config)

    os.makedirs(f"output/{experiment_hash}", exist_ok=True)

    with open(f"output/{experiment_hash}/payload.json", "w") as f:
        json.dump({}, f)
