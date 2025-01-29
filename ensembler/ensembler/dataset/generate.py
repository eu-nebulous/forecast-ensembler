import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules
from influxdb_client.domain.write_precision import WritePrecision

# Add sleep import
from time import sleep

influx_url = f"http://{os.environ.get('INFLUXDB_HOSTNAME', 'localhost')}:{os.environ.get('INFLUXDB_PORT', '8086')}"
influx_token = os.environ.get("INFLUXDB_TOKEN")
influx_org = os.environ.get("INFLUXDB_ORG", "my-org")

predictions_bucket = "nebulous_abc_predicted_bucket"
real_bucket = "nebulous_abc_bucket"

client = InfluxDBClient(
    url=influx_url,
    token=influx_token,
    org=influx_org
)

# Make sure we use the SYNCHRONOUS write options:
write_api = client.write_api(write_options=SYNCHRONOUS)
buckets_api = client.buckets_api()


def ensure_bucket_exists(bucket_name: str, org_name: str):
    existing = buckets_api.find_bucket_by_name(bucket_name)
    if existing is None:
        print(f"Bucket '{bucket_name}' not found. Creating...")
        retention_rules = BucketRetentionRules(type="expire", every_seconds=3600 * 24 * 30)
        buckets_api.create_bucket(
            bucket_name=bucket_name,
            org=org_name,
            retention_rules=retention_rules
        )
    else:
        print(f"Bucket '{bucket_name}' already exists. Skipping creation.")


ensure_bucket_exists(predictions_bucket, influx_org)
ensure_bucket_exists(real_bucket, influx_org)

# Reduced range & frequency to avoid large data
start_days_ago = 1
freq = "2h"

end_time = datetime.utcnow()
start_time = end_time - timedelta(days=start_days_ago)
date_range = pd.date_range(start=start_time, end=end_time, freq=freq)

measurement_name = "AccumulatedSecondsPendingRequests"
application_name = "abc"

# Generate data
predicted_values = np.sin(np.linspace(0, 10, len(date_range))) * 50 + 100
predicted_values += np.random.normal(0, 5, size=len(date_range))
real_values = predicted_values + np.random.normal(0, 10, size=len(date_range)) + 5

prediction_points = []
for timestamp, value in zip(date_range, predicted_values):
    p = (
        Point(measurement_name)
        .tag("application_name", application_name)
        .tag("forecaster", "lstm")
        .tag("level", "int32(1)")
        .field("metricValue", float(value))
        .time(timestamp)
    )
    prediction_points.append(p)

for timestamp, value in zip(date_range, predicted_values):
    p = (
        Point(measurement_name)
        .tag("application_name", application_name)
        .tag("forecaster", "exponentialsmoothing")
        .tag("level", "int32(1)")
        .field("metricValue", float(value))
        .time(timestamp)
    )
    prediction_points.append(p)

real_points = []
for timestamp, value in zip(date_range, real_values):
    p = (
        Point(measurement_name)
        .tag("application_name", application_name)
        .tag("level", "int32(1)")
        .field("metricValue", float(value))
        .time(timestamp)
    )
    real_points.append(p)

try:
    print(f"Writing {len(prediction_points)} prediction points...")
    write_api.write(
        bucket=predictions_bucket,
        org=influx_org,
        record=prediction_points,
        write_precision=WritePrecision.S
    )

    print(f"Writing {len(real_points)} real data points...")
    write_api.write(
        bucket=real_bucket,
        org=influx_org,
        record=real_points,
        write_precision=WritePrecision.S
    )

    print("Data generation complete (synchronous writes)!")

except Exception as e:
    print("Error writing data:", e)

finally:
    # Sleep to ensure all async processes complete (hotfix)
    sleep(10)
    client.close()
