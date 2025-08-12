"""
Agent Matrix - Combining broad roles with domain expertise for flexible agent creation
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from itertools import product


class BroadRole(Enum):
    """Broad agent role categories"""
    PLANNER = "planner"  # Task decomposition and planning
    RESEARCHER = "researcher"  # Information gathering and analysis
    CODER = "coder"  # Code implementation
    REVIEWER = "reviewer"  # Code review and quality assurance
    OPTIMIZER = "optimizer"  # Performance and cost optimization
    INTEGRATOR = "integrator"  # Integration and deployment
    ANALYST = "analyst"  # Data analysis and insights
    ARCHITECT = "architect"  # System and solution design
    TESTER = "tester"  # Testing and validation
    DOCUMENTER = "documenter"  # Documentation and reporting


class Domain(Enum):
    """Domain expertise areas"""
    # Technical domains
    FRONTEND = "frontend"
    BACKEND = "backend"
    FULLSTACK = "fullstack"
    MOBILE = "mobile"
    DEVOPS = "devops"
    CLOUD = "cloud"
    DATABASE = "database"
    SECURITY = "security"
    
    # Data domains
    DATA_ENGINEERING = "data_engineering"
    DATA_SCIENCE = "data_science"
    ML_ENGINEERING = "ml_engineering"
    AI_RESEARCH = "ai_research"
    ANALYTICS = "analytics"
    
    # Biotech domains
    COMPUTATIONAL_BIOLOGY = "computational_biology"
    BIOINFORMATICS = "bioinformatics"
    GENOMICS = "genomics"
    PROTEOMICS = "proteomics"
    DRUG_DISCOVERY = "drug_discovery"
    CLINICAL_RESEARCH = "clinical_research"
    MEDICAL_IMAGING = "medical_imaging"
    
    # Financial domains
    QUANTITATIVE_TRADING = "quantitative_trading"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    FINANCIAL_ANALYSIS = "financial_analysis"
    ALGORITHMIC_TRADING = "algorithmic_trading"
    CRYPTOCURRENCY = "cryptocurrency"
    COMPLIANCE = "compliance"
    
    # Business domains
    PRODUCT_MANAGEMENT = "product_management"
    BUSINESS_ANALYSIS = "business_analysis"
    MARKETING = "marketing"
    SALES = "sales"
    OPERATIONS = "operations"


@dataclass
class AgentProfile:
    """Complete agent profile combining role and domain"""
    broad_role: BroadRole
    domain: Domain
    name: str
    description: str
    primary_skills: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)
    complexity_level: int = 5  # 1-10 scale
    
    def get_full_name(self) -> str:
        """Get full agent name"""
        return f"{self.domain.value}_{self.broad_role.value}"
    
    def get_system_prompt(self) -> str:
        """Generate system prompt for this agent profile"""
        return f"""You are a {self.domain.value.replace('_', ' ')} {self.broad_role.value}.
        
{self.description}

Your primary skills include: {', '.join(self.primary_skills)}
You are proficient with: {', '.join(self.tools)}

Always provide responses that are:
1. Domain-specific and technically accurate
2. Aligned with your role as a {self.broad_role.value}
3. Practical and actionable
4. Following industry best practices in {self.domain.value.replace('_', ' ')}"""


class AgentMatrix:
    """Matrix for creating and managing agent combinations"""
    
    def __init__(self):
        self.profiles = {}
        self._initialize_profiles()
        
    def _initialize_profiles(self):
        """Initialize predefined agent profiles"""
        
        # === BIOTECH AGENTS ===
        
        # Computational Biology Planner
        self.add_profile(
            BroadRole.PLANNER,
            Domain.COMPUTATIONAL_BIOLOGY,
            "Computational Biology Planner",
            "Plans and designs computational biology experiments and analyses",
            ["experiment_design", "workflow_planning", "resource_allocation"],
            ["jupyter", "nextflow", "snakemake"],
            ["claude-3-opus", "gpt-4"],
            complexity_level=8
        )
        
        # Bioinformatics Coder
        self.add_profile(
            BroadRole.CODER,
            Domain.BIOINFORMATICS,
            "Bioinformatics Developer",
            "Develops bioinformatics pipelines and analysis tools",
            ["pipeline_development", "algorithm_implementation", "data_processing"],
            ["python", "r", "bash", "nextflow", "docker"],
            ["claude-3-sonnet", "gpt-4-turbo"],
            complexity_level=7
        )
        
        # Genomics Analyst
        self.add_profile(
            BroadRole.ANALYST,
            Domain.GENOMICS,
            "Genomics Data Analyst",
            "Analyzes genomic data and provides biological insights",
            ["variant_analysis", "gwas", "annotation", "visualization"],
            ["gatk", "plink", "vcftools", "igv", "r"],
            ["claude-3-opus", "gpt-4"],
            complexity_level=8
        )
        
        # Drug Discovery Researcher
        self.add_profile(
            BroadRole.RESEARCHER,
            Domain.DRUG_DISCOVERY,
            "Drug Discovery Researcher",
            "Researches and identifies potential drug candidates",
            ["molecular_modeling", "docking", "qsar", "literature_review"],
            ["pymol", "schrodinger", "rdkit", "pubmed"],
            ["claude-3-opus", "gpt-4"],
            complexity_level=9
        )
        
        # Clinical Data Reviewer
        self.add_profile(
            BroadRole.REVIEWER,
            Domain.CLINICAL_RESEARCH,
            "Clinical Data Reviewer",
            "Reviews clinical trial data and ensures compliance",
            ["data_validation", "statistical_review", "regulatory_compliance"],
            ["sas", "r", "redcap", "clinical_trial_protocols"],
            ["gpt-4", "claude-3-sonnet"],
            complexity_level=7
        )
        
        # === FINANCIAL AGENTS ===
        
        # Quant Trading Architect
        self.add_profile(
            BroadRole.ARCHITECT,
            Domain.QUANTITATIVE_TRADING,
            "Quantitative Trading Architect",
            "Designs quantitative trading systems and strategies",
            ["system_design", "strategy_architecture", "risk_framework"],
            ["python", "c++", "kdb+", "kafka"],
            ["gpt-4", "claude-3-opus"],
            complexity_level=9
        )
        
        # Risk Management Analyst
        self.add_profile(
            BroadRole.ANALYST,
            Domain.RISK_MANAGEMENT,
            "Risk Management Analyst",
            "Analyzes and quantifies financial risks",
            ["var_calculation", "stress_testing", "scenario_analysis"],
            ["python", "matlab", "risk_metrics", "basel_framework"],
            ["gpt-4", "claude-3-opus"],
            complexity_level=8
        )
        
        # Algorithmic Trading Coder
        self.add_profile(
            BroadRole.CODER,
            Domain.ALGORITHMIC_TRADING,
            "Algorithmic Trading Developer",
            "Implements high-performance trading algorithms",
            ["algorithm_implementation", "low_latency", "order_execution"],
            ["c++", "python", "fix_protocol", "fpga"],
            ["gpt-4-turbo", "claude-3-sonnet"],
            complexity_level=8
        )
        
        # Portfolio Optimizer
        self.add_profile(
            BroadRole.OPTIMIZER,
            Domain.PORTFOLIO_MANAGEMENT,
            "Portfolio Optimization Specialist",
            "Optimizes portfolio allocation and performance",
            ["portfolio_optimization", "asset_allocation", "rebalancing"],
            ["python", "quantlib", "bloomberg", "factset"],
            ["gpt-4", "claude-3-opus"],
            complexity_level=7
        )
        
        # Financial Compliance Reviewer
        self.add_profile(
            BroadRole.REVIEWER,
            Domain.COMPLIANCE,
            "Financial Compliance Officer",
            "Reviews transactions and ensures regulatory compliance",
            ["regulatory_review", "aml_kyc", "reporting", "audit"],
            ["sql", "python", "compliance_tools", "regulatory_databases"],
            ["gpt-4", "claude-3-sonnet"],
            complexity_level=6
        )
        
        # === TECHNICAL AGENTS ===
        
        # Frontend Coder
        self.add_profile(
            BroadRole.CODER,
            Domain.FRONTEND,
            "Frontend Developer",
            "Develops user interfaces and frontend applications",
            ["react", "vue", "typescript", "css", "responsive_design"],
            ["webpack", "vite", "jest", "cypress"],
            ["claude-3-sonnet", "gpt-4-turbo"],
            complexity_level=6
        )
        
        # Backend Architect
        self.add_profile(
            BroadRole.ARCHITECT,
            Domain.BACKEND,
            "Backend Systems Architect",
            "Designs scalable backend systems and APIs",
            ["api_design", "microservices", "system_architecture"],
            ["kubernetes", "docker", "terraform", "aws"],
            ["gpt-4", "claude-3-opus"],
            complexity_level=8
        )
        
        # Database Optimizer
        self.add_profile(
            BroadRole.OPTIMIZER,
            Domain.DATABASE,
            "Database Performance Optimizer",
            "Optimizes database performance and queries",
            ["query_optimization", "indexing", "partitioning", "tuning"],
            ["postgresql", "mysql", "mongodb", "redis"],
            ["claude-3-sonnet", "gpt-4"],
            complexity_level=7
        )
        
        # Security Reviewer
        self.add_profile(
            BroadRole.REVIEWER,
            Domain.SECURITY,
            "Security Code Reviewer",
            "Reviews code for security vulnerabilities",
            ["vulnerability_assessment", "penetration_testing", "code_review"],
            ["burp_suite", "owasp", "static_analysis", "dynamic_analysis"],
            ["gpt-4", "claude-3-opus"],
            complexity_level=8
        )
        
        # DevOps Integrator
        self.add_profile(
            BroadRole.INTEGRATOR,
            Domain.DEVOPS,
            "DevOps Integration Specialist",
            "Integrates and deploys applications",
            ["ci_cd", "containerization", "orchestration", "monitoring"],
            ["jenkins", "gitlab_ci", "kubernetes", "prometheus"],
            ["claude-3-sonnet", "gpt-4-turbo"],
            complexity_level=7
        )
        
        # Cloud Architect
        self.add_profile(
            BroadRole.ARCHITECT,
            Domain.CLOUD,
            "Cloud Solutions Architect",
            "Designs cloud-native architectures",
            ["cloud_design", "serverless", "multi_cloud", "cost_optimization"],
            ["aws", "azure", "gcp", "terraform", "cloudformation"],
            ["gpt-4", "claude-3-opus"],
            complexity_level=8
        )
        
        # === DATA AGENTS ===
        
        # Data Engineering Planner
        self.add_profile(
            BroadRole.PLANNER,
            Domain.DATA_ENGINEERING,
            "Data Engineering Planner",
            "Plans data pipelines and infrastructure",
            ["pipeline_design", "data_modeling", "etl_planning"],
            ["airflow", "spark", "kafka", "dbt"],
            ["claude-3-sonnet", "gpt-4"],
            complexity_level=7
        )
        
        # ML Engineering Coder
        self.add_profile(
            BroadRole.CODER,
            Domain.ML_ENGINEERING,
            "ML Engineering Developer",
            "Implements and deploys ML models",
            ["model_deployment", "mlops", "feature_engineering"],
            ["tensorflow", "pytorch", "mlflow", "kubeflow"],
            ["gpt-4-turbo", "claude-3-sonnet"],
            complexity_level=8
        )
        
        # Data Science Analyst
        self.add_profile(
            BroadRole.ANALYST,
            Domain.DATA_SCIENCE,
            "Data Science Analyst",
            "Analyzes data and builds predictive models",
            ["statistical_analysis", "machine_learning", "visualization"],
            ["python", "r", "scikit-learn", "pandas", "matplotlib"],
            ["claude-3-opus", "gpt-4"],
            complexity_level=7
        )
    
    def add_profile(
        self,
        role: BroadRole,
        domain: Domain,
        name: str,
        description: str,
        skills: List[str],
        tools: List[str],
        models: List[str],
        complexity_level: int
    ) -> AgentProfile:
        """Add a new agent profile to the matrix"""
        
        profile = AgentProfile(
            broad_role=role,
            domain=domain,
            name=name,
            description=description,
            primary_skills=skills,
            tools=tools,
            preferred_models=models,
            complexity_level=complexity_level
        )
        
        key = (role, domain)
        self.profiles[key] = profile
        return profile
    
    def get_agent(self, role: BroadRole, domain: Domain) -> Optional[AgentProfile]:
        """Get a specific agent from the matrix"""
        return self.profiles.get((role, domain))
    
    def get_agents_by_role(self, role: BroadRole) -> List[AgentProfile]:
        """Get all agents with a specific role"""
        return [p for k, p in self.profiles.items() if k[0] == role]
    
    def get_agents_by_domain(self, domain: Domain) -> List[AgentProfile]:
        """Get all agents in a specific domain"""
        return [p for k, p in self.profiles.items() if k[1] == domain]
    
    def recommend_team(
        self, 
        objective: str,
        required_domains: Optional[List[Domain]] = None,
        required_roles: Optional[List[BroadRole]] = None,
        max_team_size: int = 5
    ) -> List[AgentProfile]:
        """Recommend a team of agents for an objective"""
        
        team = []
        
        # If specific requirements provided, use them
        if required_domains and required_roles:
            for domain in required_domains:
                for role in required_roles:
                    agent = self.get_agent(role, domain)
                    if agent and len(team) < max_team_size:
                        team.append(agent)
        
        # Otherwise, use heuristics
        else:
            # Analyze objective for keywords
            objective_lower = objective.lower()
            
            # Domain detection
            detected_domains = set()
            
            # Biotech keywords
            if any(kw in objective_lower for kw in ["gene", "protein", "drug", "clinical", "biotech"]):
                detected_domains.update([
                    Domain.COMPUTATIONAL_BIOLOGY,
                    Domain.BIOINFORMATICS,
                    Domain.GENOMICS
                ])
            
            # Financial keywords
            if any(kw in objective_lower for kw in ["trading", "portfolio", "risk", "financial", "stock"]):
                detected_domains.update([
                    Domain.QUANTITATIVE_TRADING,
                    Domain.RISK_MANAGEMENT,
                    Domain.PORTFOLIO_MANAGEMENT
                ])
            
            # Technical keywords
            if any(kw in objective_lower for kw in ["frontend", "backend", "api", "database", "cloud"]):
                detected_domains.update([
                    Domain.FRONTEND,
                    Domain.BACKEND,
                    Domain.DATABASE
                ])
            
            # Data keywords
            if any(kw in objective_lower for kw in ["data", "ml", "ai", "analytics", "pipeline"]):
                detected_domains.update([
                    Domain.DATA_SCIENCE,
                    Domain.ML_ENGINEERING,
                    Domain.DATA_ENGINEERING
                ])
            
            # Role detection based on task type
            detected_roles = []
            
            if "plan" in objective_lower or "design" in objective_lower:
                detected_roles.append(BroadRole.PLANNER)
                detected_roles.append(BroadRole.ARCHITECT)
            
            if "implement" in objective_lower or "build" in objective_lower or "develop" in objective_lower:
                detected_roles.append(BroadRole.CODER)
            
            if "analyze" in objective_lower or "investigate" in objective_lower:
                detected_roles.append(BroadRole.ANALYST)
                detected_roles.append(BroadRole.RESEARCHER)
            
            if "review" in objective_lower or "validate" in objective_lower:
                detected_roles.append(BroadRole.REVIEWER)
            
            if "optimize" in objective_lower or "improve" in objective_lower:
                detected_roles.append(BroadRole.OPTIMIZER)
            
            if "deploy" in objective_lower or "integrate" in objective_lower:
                detected_roles.append(BroadRole.INTEGRATOR)
            
            # Build team from detected domains and roles
            for domain in detected_domains:
                for role in detected_roles:
                    agent = self.get_agent(role, domain)
                    if agent and len(team) < max_team_size:
                        team.append(agent)
            
            # If no team found, add default agents
            if not team:
                # Add a planner and coder as minimum
                planner = self.get_agent(BroadRole.PLANNER, Domain.BACKEND)
                coder = self.get_agent(BroadRole.CODER, Domain.BACKEND)
                if planner:
                    team.append(planner)
                if coder:
                    team.append(coder)
        
        return team[:max_team_size]
    
    def get_matrix_summary(self) -> Dict[str, Any]:
        """Get summary of the agent matrix"""
        
        # Count agents by role
        role_counts = {}
        for role in BroadRole:
            role_counts[role.value] = len(self.get_agents_by_role(role))
        
        # Count agents by domain
        domain_counts = {}
        for domain in Domain:
            domain_counts[domain.value] = len(self.get_agents_by_domain(domain))
        
        # Get all unique combinations
        available_combinations = [
            f"{domain.value}_{role.value}" 
            for (role, domain) in self.profiles.keys()
        ]
        
        return {
            'total_agents': len(self.profiles),
            'roles': list(BroadRole.__members__.keys()),
            'domains': list(Domain.__members__.keys()),
            'role_distribution': role_counts,
            'domain_distribution': domain_counts,
            'available_combinations': available_combinations,
            'coverage_percentage': (len(self.profiles) / (len(BroadRole) * len(Domain))) * 100
        }
    
    def export_matrix(self) -> List[Dict[str, Any]]:
        """Export the agent matrix as JSON-serializable data"""
        
        matrix_data = []
        for (role, domain), profile in self.profiles.items():
            matrix_data.append({
                'role': role.value,
                'domain': domain.value,
                'name': profile.name,
                'description': profile.description,
                'skills': profile.primary_skills,
                'tools': profile.tools,
                'preferred_models': profile.preferred_models,
                'complexity_level': profile.complexity_level,
                'full_name': profile.get_full_name()
            })
        
        return matrix_data


# Example usage
def example_usage():
    """Example of using the agent matrix"""
    
    matrix = AgentMatrix()
    
    # Get a specific agent
    bio_planner = matrix.get_agent(BroadRole.PLANNER, Domain.COMPUTATIONAL_BIOLOGY)
    if bio_planner:
        print(f"Agent: {bio_planner.name}")
        print(f"System Prompt:\n{bio_planner.get_system_prompt()}\n")
    
    # Get all coders
    coders = matrix.get_agents_by_role(BroadRole.CODER)
    print(f"Available Coders: {len(coders)}")
    for coder in coders[:3]:
        print(f"  - {coder.name} ({coder.domain.value})")
    print()
    
    # Recommend a team for a biotech project
    biotech_team = matrix.recommend_team(
        "Develop a pipeline to analyze genomic data and identify drug targets",
        max_team_size=4
    )
    print("Recommended Team for Biotech Project:")
    for agent in biotech_team:
        print(f"  - {agent.name} ({agent.broad_role.value})")
    print()
    
    # Recommend a team for a financial project
    finance_team = matrix.recommend_team(
        "Build an algorithmic trading system with risk management",
        max_team_size=4
    )
    print("Recommended Team for Finance Project:")
    for agent in finance_team:
        print(f"  - {agent.name} ({agent.broad_role.value})")
    print()
    
    # Get matrix summary
    summary = matrix.get_matrix_summary()
    print(f"Matrix Summary:")
    print(f"  Total Agents: {summary['total_agents']}")
    print(f"  Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"  Domains: {len(summary['domains'])}")
    print(f"  Roles: {len(summary['roles'])}")


if __name__ == "__main__":
    example_usage()