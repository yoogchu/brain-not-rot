# Chapter 30: Infrastructure as Code

## Why Infrastructure as Code?

Without IaC:

```
Engineer: "I deployed to production last night"
Manager: "What changed?"
Engineer: "Uh... I clicked around AWS console, changed some settings..."
Manager: "Which settings?"
Engineer: "I don't remember. But it's working now."

[Two weeks later]
New Engineer: "I need to replicate the prod environment"
Result: 3 days of manual clicking, configuration drift, subtle bugs
```

Infrastructure as Code (IaC) solves:
- **Reproducibility**: Exact same infrastructure every time
- **Version control**: Git history of all infrastructure changes
- **Code review**: PRs for infrastructure changes
- **Disaster recovery**: Rebuild everything from scratch in hours, not weeks
- **Documentation**: The code IS the documentation

Manual infrastructure management doesn't scale. One typo in a security group rule can expose your database to the internet. IaC catches this in code review.

---

## Declarative vs Imperative

### The Fundamental Difference

**Imperative:** Tell the system HOW to build infrastructure

```python
# Imperative (script-like)
def create_infrastructure():
    vpc = create_vpc("10.0.0.0/16")
    subnet = create_subnet(vpc.id, "10.0.1.0/24")
    instance = create_instance(subnet.id, "t3.medium")
    if not security_group_exists("web-sg"):
        create_security_group("web-sg", vpc.id)
    attach_security_group(instance.id, "web-sg")
```

**Declarative:** Tell the system WHAT you want, it figures out HOW

```hcl
# Declarative (Terraform)
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
}

resource "aws_instance" "web" {
  ami           = "ami-123456"
  instance_type = "t3.medium"
  subnet_id     = aws_subnet.public.id
}
```

**What happens on second run:**
- Imperative: Creates everything again → duplicates, errors
- Declarative: Compares desired state to actual state → no changes needed

```
Declarative workflow:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Desired State   │    │  Current State   │    │  Execution Plan  │
│  (your .tf code) │───►│  (real AWS)      │───►│  (what to change)│
└──────────────────┘    └──────────────────┘    └──────────────────┘

Plan says:
+ Create: New resources
~ Update: Modified resources
- Delete: Resources to remove
```

**When to use declarative:** Most infrastructure (preferred)
**When to use imperative:** Complex migrations, one-off scripts, conditional logic too complex for declarative

---

## Terraform Fundamentals

### The Problem

You need to manage cloud infrastructure across AWS, GCP, Azure. Each has different CLI tools, APIs, and concepts. Learning curves multiply.

### How It Works

Terraform provides a unified syntax (HCL - HashiCorp Configuration Language) that works across providers.

```
┌─────────────────────────────────────────────────┐
│          Terraform Core                          │
│          (reads .tf files, plans changes)        │
└─────────────────────────────────────────────────┘
         │               │              │
         ▼               ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│AWS Provider  │  │GCP Provider  │  │Azure Provider│
└──────────────┘  └──────────────┘  └──────────────┘
         │               │              │
         ▼               ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  AWS API     │  │  GCP API     │  │  Azure API   │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Basic Terraform File Structure

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Define resources
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name        = "${var.environment}-vpc"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_subnet" "public" {
  count = length(var.availability_zones)

  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.environment}-public-${count.index}"
  }
}

# Outputs
output "vpc_id" {
  value       = aws_vpc.main.id
  description = "The ID of the VPC"
}
```

```hcl
# variables.tf
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod"
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}
```

### Terraform Workflow

```python
# Simulating Terraform's workflow in Python for understanding

class TerraformWorkflow:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.desired_state = {}
        self.current_state = {}
        self.plan = []

    def init(self):
        """Initialize providers and backend"""
        # Download provider plugins
        # Configure remote state backend
        print("Initializing Terraform...")
        print("- Downloading AWS provider v5.0")
        print("- Configuring S3 backend")

    def parse_config(self):
        """Parse .tf files into desired state"""
        # Read all .tf files
        # Build resource graph
        self.desired_state = {
            "aws_vpc.main": {"cidr_block": "10.0.0.0/16"},
            "aws_subnet.public[0]": {"vpc_id": "aws_vpc.main.id", "cidr_block": "10.0.0.0/24"}
        }

    def read_current_state(self):
        """Query actual infrastructure state"""
        # Read terraform.tfstate
        # Or query cloud provider APIs
        self.current_state = {
            "aws_vpc.main": {"cidr_block": "10.0.0.0/16"},  # Exists, unchanged
            # aws_subnet.public[0] doesn't exist yet
        }

    def plan(self):
        """Calculate diff between desired and current"""
        for resource, config in self.desired_state.items():
            if resource not in self.current_state:
                self.plan.append(("CREATE", resource, config))
            elif self.current_state[resource] != config:
                self.plan.append(("UPDATE", resource, config))

        for resource in self.current_state:
            if resource not in self.desired_state:
                self.plan.append(("DELETE", resource, None))

        # Print plan
        print("Terraform will perform the following actions:")
        for action, resource, config in self.plan:
            if action == "CREATE":
                print(f"  + {resource}")
            elif action == "UPDATE":
                print(f"  ~ {resource}")
            elif action == "DELETE":
                print(f"  - {resource}")

    def apply(self):
        """Execute the plan"""
        for action, resource, config in self.plan:
            if action == "CREATE":
                self.create_resource(resource, config)
            elif action == "UPDATE":
                self.update_resource(resource, config)
            elif action == "DELETE":
                self.delete_resource(resource)

        # Update state file
        self.save_state()

# Usage
tf = TerraformWorkflow("./terraform")
tf.init()
tf.parse_config()
tf.read_current_state()
tf.plan()
# Review plan, then:
tf.apply()
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Multi-cloud | Works across AWS, GCP, Azure, etc. | Provider-specific quirks remain |
| State management | Tracks what's deployed | State file corruption can be catastrophic |
| Declarative | Idempotent, predictable | Complex conditional logic is harder |
| Ecosystem | Huge module registry | Module quality varies widely |

**When to use:** Most cloud infrastructure projects, especially multi-cloud or multi-account
**When NOT to use:** Kubernetes resources (use Helm/Kustomize), very dynamic infrastructure, application config

---

## State Management

### The Problem

Terraform needs to know what infrastructure exists to calculate diffs. Where does it store this?

**terraform.tfstate file:**
```json
{
  "version": 4,
  "terraform_version": "1.5.0",
  "resources": [
    {
      "type": "aws_vpc",
      "name": "main",
      "instances": [
        {
          "attributes": {
            "id": "vpc-0123456789abcdef0",
            "cidr_block": "10.0.0.0/16"
          }
        }
      ]
    }
  ]
}
```

This file maps your Terraform resources to real cloud resource IDs.

### Local State Problems

```
Developer A's state file:
  aws_instance.web → i-111111

Developer B's state file:
  aws_instance.web → i-222222

Both run terraform apply → duplicate instances!
```

### Remote State Solution

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/vpc/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"  # For locking
  }
}
```

**Remote state with locking:**

```
┌──────────────────────────────────────────────────┐
│             S3 Bucket                             │
│  production/vpc/terraform.tfstate                 │
│  production/eks/terraform.tfstate                 │
│  staging/vpc/terraform.tfstate                    │
└──────────────────────────────────────────────────┘
         ▲                          ▲
         │                          │
    ┌────────┐                 ┌────────┐
    │ Dev A  │                 │ Dev B  │
    │ Locked │                 │ Waiting│
    └────────┘                 └────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│          DynamoDB Lock Table                      │
│  LockID: production/vpc/terraform.tfstate         │
│  Owner: DevA                                      │
│  Created: 2024-01-15T10:30:00Z                   │
└──────────────────────────────────────────────────┘
```

### State Locking Implementation

```python
# Conceptual implementation of state locking

import time
import json
from datetime import datetime

class TerraformStateLock:
    def __init__(self, dynamodb_table, state_key):
        self.table = dynamodb_table
        self.state_key = state_key
        self.lock_id = f"{state_key}-{os.getpid()}"

    def acquire_lock(self, timeout=300):
        """Try to acquire lock, wait if locked by someone else"""
        start = time.time()

        while time.time() - start < timeout:
            try:
                # Attempt atomic write if lock doesn't exist
                self.table.put_item(
                    Item={
                        'LockID': self.state_key,
                        'Owner': self.lock_id,
                        'Created': datetime.now().isoformat(),
                        'Info': json.dumps({'Path': self.state_key})
                    },
                    ConditionExpression='attribute_not_exists(LockID)'
                )
                return True  # Lock acquired
            except ConditionalCheckFailedException:
                # Lock exists, check if stale
                existing = self.table.get_item(Key={'LockID': self.state_key})
                created = datetime.fromisoformat(existing['Created'])

                if (datetime.now() - created).seconds > 3600:
                    # Stale lock (>1 hour old), force release
                    print("Warning: Breaking stale lock")
                    self.force_release()
                else:
                    # Active lock, wait
                    print(f"Waiting for lock held by {existing['Owner']}")
                    time.sleep(5)

        raise TimeoutError("Could not acquire state lock")

    def release_lock(self):
        """Release lock"""
        self.table.delete_item(
            Key={'LockID': self.state_key},
            ConditionExpression='Owner = :owner',
            ExpressionAttributeValues={':owner': self.lock_id}
        )

    def force_release(self):
        """Force release (use with caution!)"""
        self.table.delete_item(Key={'LockID': self.state_key})

# Usage
lock = TerraformStateLock(dynamodb, "production/vpc/terraform.tfstate")
lock.acquire_lock()
try:
    # Run terraform apply
    apply_infrastructure_changes()
finally:
    lock.release_lock()
```

**State file best practices:**
- Never edit state file manually (use `terraform state` commands)
- Always use remote state for teams
- Enable versioning on S3 bucket (rollback capability)
- Encrypt state at rest (contains sensitive data)
- Separate state files by environment and component

**When to use:** Always for production, team projects
**When NOT to use:** Solo learning, throwaway experiments

---

## Modules and Reusability

### The Problem

Copy-pasting Terraform code across environments leads to drift and maintenance nightmares.

```
dev/main.tf:     50 lines of VPC setup
staging/main.tf: 50 lines of VPC setup (slightly different)
prod/main.tf:    50 lines of VPC setup (different again)

Bug found in VPC setup → fix in 3 places → forgot one → production incident
```

### Modules Solution

**Module structure:**
```
modules/
  vpc/
    main.tf       # Resource definitions
    variables.tf  # Input variables
    outputs.tf    # Output values
    README.md     # Documentation
```

**modules/vpc/main.tf:**
```hcl
variable "environment" {
  type = string
}

variable "cidr_block" {
  type = string
}

variable "availability_zones" {
  type = list(string)
}

resource "aws_vpc" "main" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true

  tags = {
    Name        = "${var.environment}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.cidr_block, 8, count.index)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.environment}-private-${count.index}"
  }
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}
```

**Using the module:**
```hcl
# environments/production/main.tf
module "vpc" {
  source = "../../modules/vpc"

  environment        = "production"
  cidr_block         = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

module "eks" {
  source = "../../modules/eks"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
}
```

**Module composition:**
```
┌─────────────────────────────────────────────────┐
│              Root Module (production)            │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │ VPC Module  │─►│ EKS Module  │─►│ RDS Mod. ││
│  │             │  │ (uses VPC)  │  │(uses VPC)││
│  └─────────────┘  └─────────────┘  └──────────┘│
│         │                                        │
│         ▼                                        │
│  ┌─────────────────────────────────┐            │
│  │  Security Groups Module         │            │
│  │  (shared by EKS and RDS)        │            │
│  └─────────────────────────────────┘            │
└─────────────────────────────────────────────────┘
```

**Trade-offs:**

| Aspect | Pros | Cons |
|--------|------|------|
| Reusability | Write once, use everywhere | Over-abstraction can obscure intent |
| Versioning | Lock module versions for stability | Updating shared modules requires coordination |
| Testing | Test module independently | Integration testing is still needed |
| Composability | Build complex infra from simple pieces | Dependency chains can get complex |

**When to use:** Repeated infrastructure patterns, multi-environment deployments
**When NOT to use:** One-off resources, infrastructure still in flux

---

## Pulumi: The Programming Language Approach

### The Problem

Terraform's HCL is declarative but limited. Complex logic requires workarounds. You can't use your team's existing programming language expertise.

### How Pulumi Works

Write infrastructure code in Python, TypeScript, Go, or C#.

```python
# __main__.py
import pulumi
import pulumi_aws as aws

# Create VPC
vpc = aws.ec2.Vpc("main-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    tags={
        "Name": "main-vpc",
        "Environment": pulumi.get_stack()
    }
)

# Create subnets using loops (real Python!)
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
private_subnets = []

for i, az in enumerate(availability_zones):
    subnet = aws.ec2.Subnet(f"private-{i}",
        vpc_id=vpc.id,
        cidr_block=f"10.0.{i}.0/24",
        availability_zone=az,
        tags={"Name": f"private-{i}"}
    )
    private_subnets.append(subnet)

# Complex logic is just Python
def should_create_nat_gateway():
    config = pulumi.Config()
    env = pulumi.get_stack()
    return env == "production" or config.get_bool("enable_nat")

if should_create_nat_gateway():
    nat = aws.ec2.NatGateway("nat-gw",
        subnet_id=private_subnets[0].id,
        allocation_id=eip.id
    )

# Export outputs
pulumi.export("vpc_id", vpc.id)
pulumi.export("subnet_ids", [s.id for s in private_subnets])
```

**Advanced patterns:**
```python
# Component Resource: Reusable abstraction
class PostgresDatabase(pulumi.ComponentResource):
    def __init__(self, name, vpc_id, subnet_ids, opts=None):
        super().__init__('custom:database:Postgres', name, {}, opts)

        # Create security group
        self.security_group = aws.ec2.SecurityGroup(
            f"{name}-sg",
            vpc_id=vpc_id,
            ingress=[{
                "protocol": "tcp",
                "from_port": 5432,
                "to_port": 5432,
                "cidr_blocks": ["10.0.0.0/16"]
            }],
            opts=pulumi.ResourceOptions(parent=self)
        )

        # Create subnet group
        self.subnet_group = aws.rds.SubnetGroup(
            f"{name}-subnet-group",
            subnet_ids=subnet_ids,
            opts=pulumi.ResourceOptions(parent=self)
        )

        # Create RDS instance
        self.db = aws.rds.Instance(
            f"{name}-db",
            engine="postgres",
            instance_class="db.t3.micro",
            allocated_storage=20,
            db_subnet_group_name=self.subnet_group.name,
            vpc_security_group_ids=[self.security_group.id],
            username="admin",
            password=pulumi.Config().require_secret("db_password"),
            skip_final_snapshot=True,
            opts=pulumi.ResourceOptions(parent=self)
        )

        self.register_outputs({
            "endpoint": self.db.endpoint,
            "db_name": self.db.name
        })

# Usage
db = PostgresDatabase("app-db", vpc.id, [s.id for s in private_subnets])
pulumi.export("db_endpoint", db.db.endpoint)
```

**Pulumi state:**
- Similar to Terraform, tracks deployed resources
- Uses Pulumi Cloud (SaaS) or self-hosted backend
- State includes resource metadata and outputs

**Trade-offs:**

| Aspect | Pulumi | Terraform |
|--------|--------|-----------|
| Language | Real programming languages | HCL (domain-specific) |
| Complexity | Can handle complex logic easily | Limited to HCL primitives |
| Learning curve | Use existing language skills | Learn HCL syntax |
| Ecosystem | Smaller, growing | Massive module registry |
| State | Pulumi Cloud or S3/Azure Blob | S3/GCS/Azure Blob |
| Debugging | Standard debuggers work | `terraform console`, logging |

**When to use:** Complex infrastructure logic, teams already using Python/TypeScript
**When NOT to use:** Simple infra, need maximum ecosystem/community

---

## GitOps Workflow

### The Problem

Infrastructure changes bypass code review, get applied directly from engineer laptops, and no audit trail exists.

```
Old workflow:
Developer laptop → terraform apply → AWS
                    (no review, no audit)
```

### GitOps Solution

All infrastructure changes go through Git. CI/CD applies changes automatically.

```
Developer → Git PR → Code Review → Merge → CI/CD → terraform apply
            │         │             │        │
            │         │             │        └─ Automated, audited
            │         │             └────────── Single source of truth
            │         └──────────────────────── Review and approval
            └────────────────────────────────── Version controlled
```

**GitOps workflow:**
```
┌─────────────────────────────────────────────────┐
│                  Git Repository                  │
│  main.tf, variables.tf, modules/                │
└─────────────────────────────────────────────────┘
         │                          │
         │ Push to main             │ Pull Request
         ▼                          ▼
┌──────────────────┐      ┌──────────────────┐
│  CI/CD Pipeline  │      │  CI/CD Pipeline  │
│  (production)    │      │  (plan only)     │
└──────────────────┘      └──────────────────┘
         │                          │
         │ terraform apply          │ terraform plan
         ▼                          ▼
┌──────────────────┐      ┌──────────────────┐
│  AWS Production  │      │  Post PR comment │
└──────────────────┘      └──────────────────┘
```

**GitHub Actions example:**
```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  pull_request:
    paths:
      - 'terraform/**'
  push:
    branches:
      - main
    paths:
      - 'terraform/**'

jobs:
  terraform-plan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: 1.5.0

      - name: Terraform Init
        run: terraform init
        working-directory: ./terraform

      - name: Terraform Format Check
        run: terraform fmt -check
        working-directory: ./terraform

      - name: Terraform Validate
        run: terraform validate
        working-directory: ./terraform

      - name: Terraform Plan
        id: plan
        run: terraform plan -out=tfplan
        working-directory: ./terraform
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Post Plan to PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const output = `#### Terraform Plan
            \`\`\`
            ${{ steps.plan.outputs.stdout }}
            \`\`\`
            `;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: output
            });

  terraform-apply:
    needs: terraform-plan
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    steps:
      - uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2
        with:
          terraform_version: 1.5.0

      - name: Terraform Init
        run: terraform init
        working-directory: ./terraform

      - name: Terraform Apply
        run: terraform apply -auto-approve
        working-directory: ./terraform
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Notify Slack
        if: always()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Terraform apply ${{ job.status }} on production",
              "commit": "${{ github.sha }}"
            }
```

**ArgoCD for Kubernetes:**
- Watches Git repo for Kubernetes manifests
- Automatically syncs cluster state to match Git
- Detects drift (manual kubectl changes)

```python
# Conceptual drift detection implementation

class DriftDetector:
    def __init__(self, git_repo, cluster):
        self.git_repo = git_repo
        self.cluster = cluster

    def detect_drift(self):
        """Compare Git state to cluster state"""
        git_manifests = self.load_git_manifests()
        cluster_state = self.get_cluster_state()

        drift = []

        for resource_key, git_spec in git_manifests.items():
            cluster_spec = cluster_state.get(resource_key)

            if not cluster_spec:
                drift.append({
                    "type": "missing",
                    "resource": resource_key,
                    "message": f"{resource_key} exists in Git but not in cluster"
                })
            elif git_spec != cluster_spec:
                drift.append({
                    "type": "modified",
                    "resource": resource_key,
                    "git_spec": git_spec,
                    "cluster_spec": cluster_spec
                })

        for resource_key in cluster_state:
            if resource_key not in git_manifests:
                drift.append({
                    "type": "extra",
                    "resource": resource_key,
                    "message": f"{resource_key} exists in cluster but not in Git"
                })

        return drift

    def reconcile(self, drift):
        """Fix drift to match Git"""
        for item in drift:
            if item["type"] == "missing":
                self.create_resource(item["resource"])
            elif item["type"] == "modified":
                self.update_resource(item["resource"], item["git_spec"])
            elif item["type"] == "extra":
                # Depends on policy: delete or ignore
                if self.should_prune():
                    self.delete_resource(item["resource"])

# Usage
detector = DriftDetector(git_repo="https://github.com/org/infra", cluster=k8s_client)
drift = detector.detect_drift()

if drift:
    print(f"Found {len(drift)} drifted resources")
    detector.reconcile(drift)
else:
    print("No drift detected")
```

**When to use:** Production systems, teams >3 people, compliance requirements
**When NOT to use:** Local development, prototyping

---

## Secrets Management

### The Problem

```hcl
# DON'T DO THIS
resource "aws_db_instance" "main" {
  username = "admin"
  password = "supersecret123"  # ← Committed to Git!
}
```

Secrets in IaC code end up in:
- Git history (forever)
- Terraform state file (plain text)
- CI/CD logs
- Plan output in PR comments

### Solutions

**1. Environment variables (basic):**
```hcl
variable "db_password" {
  type      = string
  sensitive = true
}

resource "aws_db_instance" "main" {
  username = "admin"
  password = var.db_password
}
```

```bash
export TF_VAR_db_password="actual-password"
terraform apply
```

**2. AWS Secrets Manager (production):**
```hcl
# Store secret
resource "aws_secretsmanager_secret" "db_password" {
  name = "production/db/password"
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = var.initial_db_password
}

# Reference secret
data "aws_secretsmanager_secret_version" "db_password" {
  secret_id = aws_secretsmanager_secret.db_password.id
}

resource "aws_db_instance" "main" {
  username = "admin"
  password = data.aws_secretsmanager_secret_version.db_password.secret_string
}
```

**3. HashiCorp Vault (enterprise):**
```hcl
data "vault_generic_secret" "db_password" {
  path = "secret/production/db"
}

resource "aws_db_instance" "main" {
  username = "admin"
  password = data.vault_generic_secret.db_password.data["password"]
}
```

**State file encryption:**
```hcl
terraform {
  backend "s3" {
    bucket  = "my-terraform-state"
    key     = "production/terraform.tfstate"
    encrypt = true  # ← Server-side encryption

    # Use KMS for additional security
    kms_key_id = "arn:aws:kms:us-east-1:123456789:key/abc-def"
  }
}
```

**When to use:**
- Environment variables: Development, simple cases
- Secrets Manager: Production, AWS-native
- Vault: Multi-cloud, enterprise, dynamic secrets

---

## Blast Radius Control

### The Problem

```
One terraform apply command destroys entire production:
- Database (with all data)
- Load balancers
- Auto-scaling groups
- Everything

Recovery time: Unknown. Data loss: Catastrophic.
```

### Strategy 1: Separate State Files

```
terraform/
  environments/
    production/
      network/          # VPC, subnets
        backend.tf      # → production-network.tfstate
      data/             # RDS, ElastiCache
        backend.tf      # → production-data.tfstate
      compute/          # EKS, EC2
        backend.tf      # → production-compute.tfstate
```

**Blast radius now limited:**
- `terraform destroy` in network/ → Only networking changes
- Can't accidentally destroy database from compute directory

### Strategy 2: Workspaces (with caution)

```bash
terraform workspace new production
terraform workspace new staging
terraform workspace new dev

# Switch context
terraform workspace select production
```

**Warning:** Same state file, different namespaces. Less safe than separate state files.

### Strategy 3: Resource Targeting

```bash
# Apply only specific resources
terraform apply -target=aws_security_group.web

# Destroy specific resource
terraform destroy -target=aws_instance.temp_debug
```

**Use sparingly:** Breaks dependency graph, can lead to inconsistent state.

### Strategy 4: Lifecycle Protections

```hcl
resource "aws_db_instance" "production" {
  # ... configuration ...

  deletion_protection = true

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [
      password,  # Don't update if password changed elsewhere
    ]
  }
}
```

**Protection layers:**
| Layer | What it prevents |
|-------|------------------|
| `prevent_destroy` | Terraform from destroying resource |
| `deletion_protection` | Cloud provider from deleting |
| State file locking | Concurrent modifications |
| RBAC | Unauthorized applies |

---

## When NOT to Use IaC

### 1. Application Configuration

```hcl
# Bad: Application config in Terraform
resource "aws_ssm_parameter" "app_config" {
  name  = "/app/feature_flags/new_ui_enabled"
  value = "true"  # Requires terraform apply to change
}
```

**Better:** Use application config systems (LaunchDarkly, ConfigCat, environment variables)

### 2. Frequently Changing Resources

```hcl
# Bad: Managing thousands of short-lived containers
resource "aws_ecs_task_definition" "user_job_12345" {
  # Changes every few minutes as users submit jobs
}
```

**Better:** Let orchestrators (Kubernetes, ECS) handle dynamic resources

### 3. Secrets Themselves

```hcl
# Bad: Secret values in IaC
resource "aws_secretsmanager_secret_version" "api_key" {
  secret_string = "sk-abc123..."  # Don't store the actual secret
}
```

**Better:** Create the secret resource, populate value manually or via separate secret management tool

### 4. Stateful Data

```hcl
# Dangerous: Database with important data
resource "aws_db_instance" "prod" {
  skip_final_snapshot = true  # ← One typo = data loss
}
```

**Better:** IaC for infrastructure, backup/restore for data

---

## Comparison: IaC Tools

| Feature | Terraform | Pulumi | CloudFormation | CDK |
|---------|-----------|---------|----------------|-----|
| **Language** | HCL | Python/TS/Go | JSON/YAML | TypeScript/Python |
| **Cloud Support** | All major clouds | All major clouds | AWS only | AWS (primary) |
| **State Management** | Required (S3/etc) | Required (Pulumi Cloud) | AWS managed | CloudFormation |
| **Learning Curve** | Medium | Low (if know language) | High | Medium |
| **Ecosystem** | Massive | Growing | AWS native | AWS focused |
| **Open Source** | Yes | Yes (with paid features) | Proprietary | Yes |
| **Best For** | Multi-cloud, mature | Complex logic | AWS-only, simple | AWS + TypeScript teams |

---

## Key Concepts Checklist

- [ ] Explain declarative vs imperative infrastructure
- [ ] Describe Terraform workflow (init, plan, apply)
- [ ] Explain state management and locking
- [ ] Design module structure for reusability
- [ ] Compare Terraform vs Pulumi vs CloudFormation
- [ ] Implement GitOps workflow for infrastructure
- [ ] Handle secrets securely (never in code)
- [ ] Control blast radius with separate state files
- [ ] Know when NOT to use IaC

---

## Practical Insights

**State file is the source of truth:**
- Corrupted state = disaster. Always enable S3 versioning.
- Lost state? Use `terraform import` to reconstruct, painful but possible.
- Never edit state manually. Use `terraform state mv`, `terraform state rm`.

**Module versioning prevents surprises:**
```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"  # Pin major version, allow minor updates
}
```
- Without version pinning: Module author's breaking change breaks your infra.
- Too strict pinning: Miss security updates.
- Goldilocks: Pin major, review updates in staging first.

**Plan review is code review:**
```
Terraform will perform the following actions:

  # aws_security_group.web will be updated in-place
  ~ resource "aws_security_group" "web" {
      ~ ingress = [
          - {
              cidr_blocks = ["10.0.0.0/16"]  # Private subnet
              from_port   = 443
            },
          + {
              cidr_blocks = ["0.0.0.0/0"]    # ← THE ENTIRE INTERNET!
              from_port   = 443
            },
        ]
    }
```
Always review plans like code reviews. A `-` followed by `+` in security group rules? Investigate thoroughly.

**Drift happens:**
```
Cause: Engineer manually changes security group via AWS Console
Result: Terraform state out of sync with reality
Fix: terraform refresh (updates state) or terraform apply (reverts change)
```
Options:
- `import` drift into Terraform (accept manual change)
- `apply` to revert drift (enforce IaC)
- Use GitOps to prevent manual changes

**Organize by rate of change:**
```
terraform/
  foundation/     # VPCs, subnets (rarely change)
  data/           # Databases (occasionally change)
  compute/        # EKS, instances (change more often)
  applications/   # App-specific resources (frequent changes)
```
Slow-changing resources get their own state. Fast-changing in separate state. This minimizes blast radius and plan noise.

**Count vs for_each:**
```hcl
# Bad: Using count
resource "aws_instance" "web" {
  count = 3  # Creates web[0], web[1], web[2]
}
# Problem: Deleting web[1] causes web[2] to become web[1], triggering replacement

# Good: Using for_each
resource "aws_instance" "web" {
  for_each = toset(["web-1", "web-2", "web-3"])
  # Creates web["web-1"], web["web-2"], web["web-3"]
}
# Deleting web-2 only affects that resource
```

**Testing infrastructure code:**
- Unit tests: Validate HCL syntax, check outputs (`terraform validate`)
- Integration tests: Deploy to test account, verify resources exist
- Contract tests: Ensure modules meet expected interface
- Tools: Terratest (Go), pytest with Pulumi

Real-world test:
```python
# test_vpc_module.py
import pytest
import subprocess
import json

def test_vpc_creates_correct_subnets():
    # Run terraform init and apply
    subprocess.run(["terraform", "init"], cwd="./test-fixtures/vpc")
    subprocess.run(["terraform", "apply", "-auto-approve"], cwd="./test-fixtures/vpc")

    # Get outputs
    result = subprocess.run(
        ["terraform", "output", "-json"],
        cwd="./test-fixtures/vpc",
        capture_output=True
    )
    outputs = json.loads(result.stdout)

    # Assert
    assert len(outputs["private_subnet_ids"]["value"]) == 3
    assert outputs["vpc_cidr"]["value"] == "10.0.0.0/16"

    # Cleanup
    subprocess.run(["terraform", "destroy", "-auto-approve"], cwd="./test-fixtures/vpc")
```
