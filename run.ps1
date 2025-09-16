param (
    [string]$command = "help"
)

switch ($command) {
    "build" {
        docker-compose build
    }
    "up" {
        docker-compose up -d
    }
    "down" {
        docker-compose down
    }
    "logs-api" {
        docker logs -f churn-api
    }
    "logs-dashboard" {
        docker logs -f churn-dashboard
    }
    "api" {
        Start-Process "http://localhost:8001/docs"
    }
    "dashboard" {
        Start-Process "http://localhost:8501"
    }
    "test-latency" {
        python test_latency.py
    }
    "load-test" {
        python load_test.py
    }
    Default {
        Write-Host " Commandes disponibles :"
        Write-Host "  ./run.ps1 build        -> build images"
        Write-Host "  ./run.ps1 up           -> start containers"
        Write-Host "  ./run.ps1 down         -> stop containers"
        Write-Host "  ./run.ps1 logs-api     -> show API logs"
        Write-Host "  ./run.ps1 logs-dashboard -> show dashboard logs"
        Write-Host "  ./run.ps1 api          -> open API docs"
        Write-Host "  ./run.ps1 dashboard    -> open Streamlit app"
        Write-Host "  ./run.ps1 test-latency -> run latency test"
        Write-Host "  ./run.ps1 load-test    -> run load test"
    }
}
