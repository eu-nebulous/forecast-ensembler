# Ensembler 

To test locally:

    cd ../../
    docker build -t ensembler -f ./deployment/ensembler/Dockerfile .
    docker run -t --env-file=deployment/ensembler/env -v <LOGS PATH>:/wd/logs --network=host ensembler

Go to local amw web cosnole: http://localhost:8161/admin

send example start_ensembler message:

    {
    "metrics":[
        {
            "metric":"MaxCPULoad",
            "level":3,
            "publish_rate":60000
        },
        {
            "metric":"MinCPULoad",
            "level":3,
            "publish_rate":50000
        }
    ],
    "models":[
        "tft",
        "nbeats",
        "gluon"
    ]
    }


send ensemble request:

    curl -i http://127.0.0.1:8000/ensemble -X POST -H 'Content-Type: application/json' -d '{"method":"BestSubset", "metric": "MaxCPULoad", "predictionTime": 1234567, "predictionsToEnsemble": {"tft": 0, "nbeats": null, "gluon": 9}}' -w '\n'




