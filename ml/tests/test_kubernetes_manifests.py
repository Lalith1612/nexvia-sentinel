"""
Automated Validation for Kubernetes Manifests and Docker Specifications (Phase 4).

Verifies:
1. Valid YAML syntax for all manifests in k8s/.
2. Deployments specify valid replicas, images, resource requests/limits, and health probes.
3. Services properly bind targetPorts and selectors.
4. PersistentVolumeClaim specifies ReadWriteOnce access mode for SQLite compatibility.
5. ConfigMap contains non-secret operational settings (no passwords or credentials).
6. Frontend Dockerfile and Nginx configuration exist and match reverse-proxy routing.
"""

from pathlib import Path
import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
K8S_DIR = REPO_ROOT / "k8s"
FRONTEND_DIR = REPO_ROOT / "frontend"


def test_k8s_directory_contents():
    """Verifies that k8s/ contains the expected manifests."""
    expected_files = {
        "configmap.yaml",
        "backend-pvc.yaml",
        "backend-deployment.yaml",
        "backend-service.yaml",
        "frontend-deployment.yaml",
        "frontend-service.yaml",
        "kustomization.yaml",
    }
    actual_files = {f.name for f in K8S_DIR.glob("*.yaml")}
    assert expected_files.issubset(actual_files), f"Missing manifests: {expected_files - actual_files}"


def test_yaml_syntax_all_manifests():
    """Verifies all YAML files in k8s/ parse cleanly without syntax errors."""
    for yf in K8S_DIR.glob("*.yaml"):
        with open(yf, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            assert len(docs) > 0, f"Empty manifest: {yf.name}"
            for doc in docs:
                assert isinstance(doc, dict), f"Malformed document in {yf.name}"


def test_backend_deployment_spec():
    """Verifies backend deployment has safe single-replica, probes, and resource limits."""
    with open(K8S_DIR / "backend-deployment.yaml", "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    spec = doc["spec"]
    # Single replica strictly required for in-memory graph state snapshot and SQLite locking
    assert spec["replicas"] == 1
    assert spec["strategy"]["type"] == "Recreate"

    container = spec["template"]["spec"]["containers"][0]
    assert container["name"] == "backend"
    assert container["ports"][0]["containerPort"] == 8000

    # Resource requests and limits
    res = container["resources"]
    assert "requests" in res and "limits" in res
    assert res["requests"]["cpu"] == "250m"
    assert res["requests"]["memory"] == "512Mi"
    assert res["limits"]["cpu"] == "1000m"
    assert res["limits"]["memory"] == "1536Mi"

    # Health probes
    liveness = container["livenessProbe"]
    assert liveness["httpGet"]["path"] == "/api/v1/health"
    assert liveness["httpGet"]["port"] == 8000
    assert liveness["initialDelaySeconds"] >= 20

    readiness = container["readinessProbe"]
    assert readiness["httpGet"]["path"] == "/api/v1/health"
    assert readiness["httpGet"]["port"] == 8000

    # Volume mount for SQLite persistence
    mounts = [vm["mountPath"] for vm in container["volumeMounts"]]
    assert "/data" in mounts


def test_backend_pvc_spec():
    """Verifies PVC uses ReadWriteOnce access mode for SQLite single-node compatibility."""
    with open(K8S_DIR / "backend-pvc.yaml", "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    spec = doc["spec"]
    assert "ReadWriteOnce" in spec["accessModes"]
    assert spec["resources"]["requests"]["storage"] == "1Gi"


def test_backend_service_spec():
    """Verifies backend ClusterIP service targets containerPort 8000."""
    with open(K8S_DIR / "backend-service.yaml", "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    spec = doc["spec"]
    assert spec["type"] == "ClusterIP"
    assert spec["selector"]["app"] == "sentinel-backend"
    assert spec["ports"][0]["port"] == 8000
    assert spec["ports"][0]["targetPort"] == 8000


def test_frontend_deployment_spec():
    """Verifies frontend deployment has probes, port 80, and memory limits."""
    with open(K8S_DIR / "frontend-deployment.yaml", "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    spec = doc["spec"]
    container = spec["template"]["spec"]["containers"][0]
    assert container["name"] == "frontend"
    assert container["ports"][0]["containerPort"] == 80

    # Probes
    assert container["livenessProbe"]["httpGet"]["path"] == "/"
    assert container["readinessProbe"]["httpGet"]["path"] == "/"

    # Resources
    res = container["resources"]
    assert "requests" in res and "limits" in res
    assert res["requests"]["memory"] == "64Mi"


def test_frontend_service_spec():
    """Verifies frontend NodePort service exposes port 80 with nodePort 30080."""
    with open(K8S_DIR / "frontend-service.yaml", "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    spec = doc["spec"]
    assert spec["type"] == "NodePort"
    assert spec["selector"]["app"] == "sentinel-frontend"
    assert spec["ports"][0]["port"] == 80
    assert spec["ports"][0]["nodePort"] == 30080


def test_configmap_no_secrets():
    """Verifies ConfigMap contains non-sensitive operational settings and zero credentials."""
    with open(K8S_DIR / "configmap.yaml", "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    data = doc["data"]
    assert "ENVIRONMENT" in data
    assert "PORT" in data
    assert "LEDGER_DB_PATH" in data
    assert "DEFAULT_WEIGHT_XGB" in data

    # Security check: ensure no passwords or API keys are embedded
    for key, val in data.items():
        assert "password" not in key.lower()
        assert "secret" not in key.lower()
        assert "token" not in key.lower()
        assert "api_key" not in key.lower()


def test_frontend_docker_and_nginx_files():
    """Verifies frontend Dockerfile and Nginx reverse-proxy configuration exist."""
    dockerfile = FRONTEND_DIR / "Dockerfile"
    nginx_conf = FRONTEND_DIR / "nginx.conf"

    assert dockerfile.exists(), "frontend/Dockerfile missing"
    assert nginx_conf.exists(), "frontend/nginx.conf missing"

    content = nginx_conf.read_text(encoding="utf-8")
    assert "proxy_pass http://sentinel-backend:8000/api/;" in content
    assert "try_files $uri $uri/ /index.html;" in content
