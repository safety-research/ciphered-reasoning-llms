import os
import psycopg2
import json
import hashlib


def compute_experiment_hash(config):
    sha1 = hashlib.sha1()
    sha1.update(json.dumps(config["experiment"], sort_keys=True))
    return sha1.hexdigest()


def save_experiment_meta(config):
    experiment_hash = compute_experiment_hash(config)

    with open(f"output/{experiment_hash}/payload.json", "r") as f:
        d_payload = json.load(f)

    conn_string = os.environ["SUPABASE_CONNECTION_URL"]

    print(f"Connecting to database\n	-> {conn_string}")

    conn = psycopg2.connect(conn_string)

    table_name = config["experiment"]["table_name"]

    d_payload = {**d_payload, **config["experiment"]}

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO %s (data, experiment_hash) VALUES (%s) ON CONFLICT (experiment_hash) DO UPDATE SET data = EXCLUDED.data, experiment_hash = EXCLUDED.experiment_hash, created_at = EXCLUDED.created_at",
        (table_name, json.dumps(d_payload), experiment_hash),
    )
