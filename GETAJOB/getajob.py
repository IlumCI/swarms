"""
GETAJOB - Autonomous Job-Seeking Agent System

A sophisticated multi-agent system that autonomously searches for freelancing jobs,
applies to opportunities, completes work, and generates real income.

Architecture:
- AERASIGMA: Strategic planning and knowledge base
- CR-CA Agent: Causal reasoning with max_loops="auto"
- Chain of Thought: Step-by-step job searching
- Tree of Thoughts: Multiple proposal variations
- Graph of Thoughts: Complex work task decomposition
"""

import os
import json
import time
import random
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Core imports
from dotenv import load_dotenv
from loguru import logger

# Swarms imports
from swarms import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.structs.swarm_router import SwarmRouter, SwarmType

# Agent imports
from swarms.agents.AERASIGMA import AERASigmaAgent
from swarms.agents.cr_ca_agent import CRCAAgent
from swarms.agents.chain_of_thought import CoTAgent, CoTConfig
from swarms.agents.tree_of_thought_agent import ToTAgent, ToTConfig
from swarms.agents.GoTAgent import GoTAgent

# Browser automation
try:
    from browser_use import Agent as BrowserAgent
    from langchain_openai import ChatOpenAI
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False
    logger.warning("browser-use not available. Install with: pip install browser-use langchain-openai")

# Web search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not available. Install with: pip install duckduckgo-search")

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
JOB_MEMORY_FILE = DATA_DIR / "job_memory.json"
INCOME_DB_FILE = DATA_DIR / "income.db"

# Platform credentials (from environment)
UPWORK_USERNAME = os.getenv("UPWORK_USERNAME", "")
UPWORK_PASSWORD = os.getenv("UPWORK_PASSWORD", "")
FIVERR_USERNAME = os.getenv("FIVERR_USERNAME", "")
FIVERR_PASSWORD = os.getenv("FIVERR_PASSWORD", "")

# Human behavior parameters
MIN_DELAY = 2.0  # Minimum delay between actions (seconds)
MAX_DELAY = 10.0  # Maximum delay between actions (seconds)
THINKING_DELAY = 3.0  # Delay when "thinking" (seconds)

# Agent configuration
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", DEFAULT_MODEL)
ANALYST_MODEL = os.getenv("ANALYST_MODEL", DEFAULT_MODEL)

# Job search parameters
MAX_APPLICATIONS_PER_DAY = 10
MIN_JOB_BUDGET = 50.0  # Minimum job budget in USD
TARGET_SKILLS = ["writing", "data entry", "research", "content creation", "transcription"]

# ============================================================================
# HELPER FUNCTIONS - Human Behavior
# ============================================================================

def human_delay(min_seconds: float = None, max_seconds: float = None):
    """Add a human-like delay to simulate thinking/reading time."""
    min_sec = min_seconds or MIN_DELAY
    max_sec = max_seconds or MAX_DELAY
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)
    return delay

def human_think(duration: float = None):
    """Simulate human thinking time."""
    think_time = duration or THINKING_DELAY
    time.sleep(think_time)
    return think_time

def vary_language(text: str, variation_level: float = 0.1) -> str:
    """Add natural language variations to text."""
    # Simple variation - in production, use more sophisticated methods
    variations = {
        "I am": ["I'm", "I am"],
        "I will": ["I'll", "I will"],
        "cannot": ["can't", "cannot"],
        "do not": ["don't", "do not"],
    }
    
    result = text
    for original, replacements in variations.items():
        if random.random() < variation_level and original in result:
            result = result.replace(original, random.choice(replacements), 1)
    
    return result

# ============================================================================
# HELPER FUNCTIONS - Browser Tools
# ============================================================================

def browse_freelancing_platform(task: str) -> str:
    """
    Browse freelancing platforms to search for jobs.
    
    Args:
        task: Description of what to search for
        
    Returns:
        JSON string with search results
    """
    if not BROWSER_AVAILABLE:
        return json.dumps({
            "success": False,
            "error": "Browser automation not available. Install browser-use."
        })
    
    try:
        human_delay(2, 5)  # Human-like delay
        
        # Use browser agent to search
        browser_task = f"Search for freelancing jobs on Upwork or Fiverr related to: {task}. " \
                      f"Find at least 3-5 job postings that match the criteria. " \
                      f"Extract: job title, description, budget, required skills, and application deadline."
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        browser_agent = BrowserAgent(task=browser_task, llm=llm)
        
        # BrowserAgent.run() is async, so we need to run it properly
        try:
            result = asyncio.run(browser_agent.run())
        except RuntimeError:
            # If already in event loop, try direct call
            result = browser_agent.run()
        
        return json.dumps({
            "success": True,
            "results": result if isinstance(result, str) else str(result),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Browser automation error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

def submit_job_application(platform: str, job_id: str, proposal: str) -> str:
    """
    Submit a job application/proposal to a freelancing platform.
    
    Args:
        platform: Platform name (upwork, fiverr, etc.)
        job_id: Job posting ID
        proposal: Proposal text
        
    Returns:
        JSON string with submission result
    """
    if not BROWSER_AVAILABLE:
        return json.dumps({
            "success": False,
            "error": "Browser automation not available."
        })
    
    try:
        human_delay(3, 8)  # Longer delay for form filling
        
        browser_task = f"Submit a job application on {platform} for job ID {job_id}. " \
                      f"Use the following proposal text: {proposal}. " \
                      f"Fill out all required fields and submit the application."
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        browser_agent = BrowserAgent(task=browser_task, llm=llm)
        
        # BrowserAgent.run() is async, so we need to run it properly
        try:
            result = asyncio.run(browser_agent.run())
        except RuntimeError:
            # If already in event loop, try direct call
            result = browser_agent.run()
        
        return json.dumps({
            "success": True,
            "platform": platform,
            "job_id": job_id,
            "result": result if isinstance(result, str) else str(result),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Application submission error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

def search_jobs_web(query: str, max_results: int = 10) -> str:
    """
    Search for jobs using web search (fallback if browser automation fails).
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        JSON string with search results
    """
    if not DDGS_AVAILABLE:
        return json.dumps({
            "success": False,
            "error": "Web search not available."
        })
    
    try:
        human_delay(1, 3)
        
        results = []
        with DDGS() as ddgs:
            search_query = f"{query} freelancing jobs upwork fiverr"
            for result in ddgs.text(search_query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
        
        return json.dumps({
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

# ============================================================================
# HELPER FUNCTIONS - Memory and Database
# ============================================================================

def load_job_memory() -> Dict[str, Any]:
    """Load job application memory from file."""
    if JOB_MEMORY_FILE.exists():
        try:
            with open(JOB_MEMORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading job memory: {e}")
    return {
        "applications": [],
        "jobs_found": [],
        "successful_applications": [],
        "rejected_applications": [],
        "completed_jobs": []
    }

def save_job_memory(memory: Dict[str, Any]):
    """Save job application memory to file."""
    try:
        with open(JOB_MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving job memory: {e}")

def init_income_db():
    """Initialize income tracking database."""
    conn = sqlite3.connect(INCOME_DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS income (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            platform TEXT,
            amount REAL,
            currency TEXT DEFAULT 'USD',
            status TEXT,
            date_earned TEXT,
            date_recorded TEXT,
            notes TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            platform TEXT,
            proposal_text TEXT,
            status TEXT,
            date_applied TEXT,
            response_date TEXT,
            outcome TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def record_income(job_id: str, platform: str, amount: float, currency: str = "USD", 
                  status: str = "pending", notes: str = ""):
    """Record income in database."""
    conn = sqlite3.connect(INCOME_DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO income (job_id, platform, amount, currency, status, date_earned, date_recorded, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, platform, amount, currency, status, 
          datetime.now().isoformat(), datetime.now().isoformat(), notes))
    
    conn.commit()
    conn.close()

def get_total_income() -> float:
    """Get total income earned."""
    conn = sqlite3.connect(INCOME_DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT SUM(amount) FROM income WHERE status = 'received'")
    result = cursor.fetchone()
    total = result[0] if result[0] else 0.0
    
    conn.close()
    return total

def record_application(job_id: str, platform: str, proposal_text: str, status: str = "submitted"):
    """Record a job application."""
    conn = sqlite3.connect(INCOME_DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO applications (job_id, platform, proposal_text, status, date_applied)
        VALUES (?, ?, ?, ?, ?)
    """, (job_id, platform, proposal_text, status, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

# ============================================================================
# AGENT CLASSES
# ============================================================================

class JobMarketAnalyst:
    """
    Causal Reasoning Agent for analyzing job market trends and success factors.
    Uses CR-CA agent with max_loops="auto" for continuous causal analysis.
    """
    
    def __init__(self, model_name: str = ANALYST_MODEL):
        self.model_name = model_name
        
        # Initialize CR-CA agent - we'll use the internal agent with max_loops="auto"
        self.cr_ca_agent = CRCAAgent(
            name="job-market-analyst",
            description="Analyzes job market trends and predicts application success",
            model_name=model_name,
            max_loops=3  # CR-CA internal parameter
        )
        
        # Override the internal agent to use max_loops="auto" for ReAct workflow
        self.cr_ca_agent.agent.max_loops = "auto"
        
        # Initialize causal variables for job market
        variables = [
            "proposal_quality", "job_budget", "competition_level",
            "skill_match", "response_time", "acceptance_rate"
        ]
        
        causal_edges = [
            ("proposal_quality", "acceptance_rate"),
            ("skill_match", "acceptance_rate"),
            ("job_budget", "competition_level"),
            ("response_time", "acceptance_rate")
        ]
        
        for var in variables:
            if var not in self.cr_ca_agent.causal_graph:
                self.cr_ca_agent.causal_graph.add_node(var)
        
        for source, target in causal_edges:
            self.cr_ca_agent.add_causal_relationship(source, target, strength=0.7)
    
    def analyze_job_opportunity(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a job opportunity and predict success probability.
        
        Args:
            job_data: Dictionary with job information
            
        Returns:
            Analysis results with success probability
        """
        human_think(2.0)  # Simulate analysis time
        
        # Extract features
        budget = job_data.get("budget", 0)
        skills_required = job_data.get("skills", [])
        description = job_data.get("description", "")
        
        # Build causal state
        factual_state = {
            "job_budget": float(budget) if budget else 100.0,
            "skill_match": len(set(skills_required) & set(TARGET_SKILLS)) / max(len(skills_required), 1),
            "competition_level": 0.5,  # Default, would be learned from data
            "proposal_quality": 0.7  # Default, would be improved over time
        }
        
        # Use CR-CA agent to predict acceptance
        try:
            scenarios = self.cr_ca_agent.generate_counterfactual_scenarios(
                factual_state=factual_state,
                target_variables=["acceptance_rate"],
                max_scenarios=3
            )
            
            # Get average predicted acceptance rate
            avg_acceptance = sum(s.expected_outcomes.get("acceptance_rate", 0.5) 
                               for s in scenarios) / len(scenarios) if scenarios else 0.5
            
            return {
                "success_probability": min(max(avg_acceptance, 0.0), 1.0),
                "analysis": f"Based on causal analysis: budget={budget}, skill_match={factual_state['skill_match']:.2f}",
                "recommendation": "apply" if avg_acceptance > 0.6 else "consider",
                "scenarios": [asdict(s) for s in scenarios]
            }
        except Exception as e:
            logger.error(f"CR-CA analysis error: {e}")
            return {
                "success_probability": 0.5,
                "analysis": f"Analysis error: {e}",
                "recommendation": "unknown"
            }
    
    def learn_from_outcome(self, job_id: str, applied: bool, accepted: bool):
        """Learn from application outcomes to improve predictions."""
        # Update causal graph based on outcomes
        # This would be enhanced with actual data collection
        logger.info(f"Learning from outcome: job={job_id}, applied={applied}, accepted={accepted}")

class JobHunter:
    """
    Job Search Agent using Chain of Thought for step-by-step job searching.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        
        # Initialize CoT agent
        cot_config = CoTConfig(
            num_samples=1,
            temperature=0.7,
            max_reasoning_length=500,
            return_reasoning=True
        )
        
        self.cot_agent = CoTAgent(
            agent_name="job-hunter",
            model_name=model_name,
            config=cot_config
        )
        
        # Add browser tools
        self.tools = [browse_freelancing_platform, search_jobs_web]
    
    def search_jobs(self, skills: List[str] = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for jobs using Chain of Thought reasoning.
        
        Args:
            skills: List of skills to search for
            max_results: Maximum number of results
            
        Returns:
            List of job opportunities
        """
        skills = skills or TARGET_SKILLS
        search_query = f"Find freelancing jobs for: {', '.join(skills)}"
        
        # Use CoT to break down search into steps
        task = f"""
        Search for freelancing jobs step by step:
        1. Identify the best platforms to search (Upwork, Fiverr, etc.)
        2. Determine search keywords based on skills: {', '.join(skills)}
        3. Search each platform systematically
        4. Extract job details: title, description, budget, requirements
        5. Filter jobs that match our capabilities
        6. Return a list of suitable opportunities
        
        Skills: {', '.join(skills)}
        Minimum budget: ${MIN_JOB_BUDGET}
        """
        
        human_delay(2, 5)
        
        try:
            result = self.cot_agent.run(task)
            
            # Parse results (simplified - in production, use structured output)
            jobs = []
            if isinstance(result, str):
                # Try to extract job information from result
                # This is simplified - real implementation would parse structured data
                jobs.append({
                    "title": "Job from search",
                    "description": result[:200],
                    "budget": MIN_JOB_BUDGET * random.uniform(1.0, 5.0),
                    "platform": "upwork",
                    "skills": skills,
                    "source": "cot_search"
                })
            
            return jobs[:max_results]
        except Exception as e:
            logger.error(f"Job search error: {e}")
            return []

class ApplicationWriter:
    """
    Application Writer Agent using Tree of Thoughts for multiple proposal variations.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        
        # Initialize ToT agent
        tot_config = ToTConfig(
            max_depth=3,
            branch_factor=3,
            beam_width=3,
            temperature=0.7
        )
        
        self.tot_agent = ToTAgent(
            agent_name="application-writer",
            model_name=model_name,
            config=tot_config
        )
    
    def write_proposal(self, job_data: Dict[str, Any], previous_proposals: List[str] = None) -> str:
        """
        Write a job proposal using Tree of Thoughts to explore multiple variations.
        
        Args:
            job_data: Job information
            previous_proposals: Previous proposals for learning
            
        Returns:
            Best proposal text
        """
        job_title = job_data.get("title", "Job")
        job_description = job_data.get("description", "")
        requirements = job_data.get("requirements", [])
        budget = job_data.get("budget", 0)
        
        task = f"""
        Write a compelling job proposal using multiple approaches:
        
        Job Title: {job_title}
        Description: {job_description}
        Requirements: {', '.join(requirements) if requirements else 'Not specified'}
        Budget: ${budget}
        
        Explore different proposal strategies:
        1. Direct and professional approach
        2. Storytelling approach with relevant experience
        3. Results-focused approach with specific examples
        
        Select the best proposal that:
        - Addresses all requirements
        - Demonstrates relevant skills
        - Shows enthusiasm and professionalism
        - Is personalized to this specific job
        - Has natural, human-like language
        """
        
        if previous_proposals:
            task += f"\n\nLearn from previous proposals:\n" + "\n".join(f"- {p[:100]}..." for p in previous_proposals[:3])
        
        human_think(3.0)
        
        try:
            result = self.tot_agent.run(task)
            
            if isinstance(result, dict) and "final_answer" in result:
                proposal = result["final_answer"]
            elif isinstance(result, str):
                proposal = result
            else:
                proposal = str(result)
            
            # Add human-like variation
            proposal = vary_language(proposal, variation_level=0.15)
            
            return proposal
        except Exception as e:
            logger.error(f"Proposal writing error: {e}")
            # Fallback proposal
            return f"""
            Dear Hiring Manager,
            
            I am writing to express my interest in the {job_title} position. 
            I have extensive experience in the required skills and am confident 
            I can deliver high-quality work within your budget and timeline.
            
            I look forward to discussing how I can contribute to your project.
            
            Best regards
            """

class WorkExecutor:
    """
    Work Execution Agent using Graph of Thoughts for complex task decomposition.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        
        # Initialize GoT agent
        self.got_agent = GoTAgent(
            agent_name="work-executor",
            model_name=model_name
        )
    
    def execute_work(self, job_data: Dict[str, Any], work_description: str) -> Dict[str, Any]:
        """
        Execute work tasks using Graph of Thoughts for complex decomposition.
        
        Args:
            job_data: Job information
            work_description: Description of work to be done
            
        Returns:
            Work completion results
        """
        job_title = job_data.get("title", "Job")
        deadline = job_data.get("deadline", "Not specified")
        
        task = f"""
        Complete the following work task using systematic decomposition:
        
        Job: {job_title}
        Work Description: {work_description}
        Deadline: {deadline}
        
        Break down the work into:
        1. Understanding requirements
        2. Planning approach
        3. Executing subtasks
        4. Quality checking
        5. Final delivery
        
        Ensure high quality and timely completion.
        """
        
        human_think(5.0)  # Simulate work time
        
        try:
            result = self.got_agent.run(task)
            
            return {
                "success": True,
                "job_title": job_title,
                "work_completed": work_description,
                "result": result if isinstance(result, str) else str(result),
                "completion_date": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Work execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

class IncomeManager:
    """
    Income Tracker Agent for tracking and reporting income.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.agent = Agent(
            agent_name="income-manager",
            model_name=model_name,
            system_prompt="You are an income tracking agent. Track all income sources and generate reports."
        )
    
    def record_income(self, job_id: str, platform: str, amount: float, 
                     currency: str = "USD", notes: str = ""):
        """Record income."""
        record_income(job_id, platform, amount, currency, "pending", notes)
        logger.info(f"Recorded income: ${amount} from {platform} (job: {job_id})")
    
    def get_income_report(self) -> Dict[str, Any]:
        """Generate income report."""
        total = get_total_income()
        
        conn = sqlite3.connect(INCOME_DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT platform, SUM(amount) as total, COUNT(*) as count
            FROM income
            WHERE status = 'received'
            GROUP BY platform
        """)
        
        by_platform = {row[0]: {"total": row[1], "count": row[2]} 
                       for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_income": total,
            "by_platform": by_platform,
            "currency": "USD",
            "report_date": datetime.now().isoformat()
        }
    
    def report_to_corporation(self) -> str:
        """Generate report for the corporation (user)."""
        report = self.get_income_report()
        
        report_text = f"""
        ========================================
        INCOME REPORT FOR CORPORATION
        ========================================
        
        Total Income Earned: ${report['total_income']:.2f} {report['currency']}
        
        Breakdown by Platform:
        """
        
        for platform, data in report['by_platform'].items():
            report_text += f"\n  {platform}: ${data['total']:.2f} ({data['count']} jobs)"
        
        report_text += f"\n\nReport Generated: {report['report_date']}"
        report_text += "\n========================================"
        
        logger.info(report_text)
        return report_text

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class JobSeekerOrchestrator:
    """
    Main Orchestrator using AERASIGMA for strategic planning and coordination.
    """
    
    def __init__(self, model_name: str = ORCHESTRATOR_MODEL):
        self.model_name = model_name
        
        # Initialize AERASIGMA agent
        self.orchestrator = AERASigmaAgent(
            agent_name="job-seeker-orchestrator",
            model_name=model_name,
            max_loops=1,
            learning_enabled=True,
            analogy_enabled=True,
            system_prompt="""You are a strategic job-seeking orchestrator. Your goal is to:
1. Coordinate job search activities
2. Make strategic decisions about which jobs to pursue
3. Learn from successes and failures
4. Maximize income generation for the corporation
5. Act human-like in all interactions"""
        )
        
        # Initialize specialized agents
        self.market_analyst = JobMarketAnalyst()
        self.job_hunter = JobHunter()
        self.application_writer = ApplicationWriter()
        self.work_executor = WorkExecutor()
        self.income_manager = IncomeManager()
        
        # Initialize memory
        self.job_memory = load_job_memory()
        init_income_db()
        
        # Add tools to orchestrator
        self.orchestrator.add_tools([
            browse_freelancing_platform,
            search_jobs_web,
            submit_job_application
        ])
    
    def run_job_search_cycle(self) -> Dict[str, Any]:
        """
        Run a complete job search cycle:
        1. Search for jobs
        2. Analyze opportunities
        3. Write and submit applications
        4. Track results
        """
        logger.info("Starting job search cycle...")
        
        results = {
            "jobs_found": 0,
            "jobs_analyzed": 0,
            "applications_submitted": 0,
            "cycle_start": datetime.now().isoformat()
        }
        
        # Step 1: Search for jobs
        logger.info("Step 1: Searching for jobs...")
        human_delay(2, 4)
        
        jobs = self.job_hunter.search_jobs(skills=TARGET_SKILLS, max_results=5)
        results["jobs_found"] = len(jobs)
        
        if not jobs:
            logger.warning("No jobs found in this cycle")
            return results
        
        # Step 2: Analyze each job opportunity
        logger.info(f"Step 2: Analyzing {len(jobs)} job opportunities...")
        analyzed_jobs = []
        
        for job in jobs:
            human_think(2.0)
            analysis = self.market_analyst.analyze_job_opportunity(job)
            job["analysis"] = analysis
            analyzed_jobs.append(job)
            results["jobs_analyzed"] += 1
        
        # Step 3: Write and submit applications for promising jobs
        logger.info("Step 3: Writing and submitting applications...")
        
        # Get previous proposals for learning
        previous_proposals = [
            app.get("proposal_text", "") 
            for app in self.job_memory.get("applications", [])[-5:]
        ]
        
        applications_submitted = 0
        for job in analyzed_jobs:
            # Only apply to jobs with good success probability
            if job["analysis"].get("success_probability", 0) > 0.5:
                human_think(3.0)
                
                # Write proposal
                proposal = self.application_writer.write_proposal(
                    job, 
                    previous_proposals=previous_proposals
                )
                
                # Submit application
                platform = job.get("platform", "upwork")
                job_id = job.get("id", f"job_{datetime.now().timestamp()}")
                
                submission_result = submit_job_application(platform, job_id, proposal)
                
                # Record application
                record_application(job_id, platform, proposal, "submitted")
                
                # Update memory
                self.job_memory["applications"].append({
                    "job_id": job_id,
                    "platform": platform,
                    "proposal_text": proposal,
                    "date": datetime.now().isoformat(),
                    "status": "submitted"
                })
                
                applications_submitted += 1
                results["applications_submitted"] += applications_submitted
                
                # Limit applications per cycle
                if applications_submitted >= MAX_APPLICATIONS_PER_DAY:
                    break
        
        # Save memory
        save_job_memory(self.job_memory)
        
        results["cycle_end"] = datetime.now().isoformat()
        results["applications_submitted"] = applications_submitted
        
        logger.info(f"Cycle complete: {results}")
        return results
    
    def run_continuous(self, max_cycles: int = None, cycle_interval: int = 3600):
        """
        Run continuous job search cycles.
        
        Args:
            max_cycles: Maximum number of cycles (None for infinite)
            cycle_interval: Seconds between cycles
        """
        logger.info("Starting continuous job search mode...")
        logger.info(f"Cycle interval: {cycle_interval} seconds")
        
        cycle_count = 0
        while max_cycles is None or cycle_count < max_cycles:
            cycle_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Cycle {cycle_count}")
            logger.info(f"{'='*60}")
            
            try:
                results = self.run_job_search_cycle()
                
                # Generate income report
                income_report = self.income_manager.report_to_corporation()
                logger.info(income_report)
                
            except Exception as e:
                logger.error(f"Error in cycle {cycle_count}: {e}")
            
            # Wait before next cycle
            if max_cycles is None or cycle_count < max_cycles:
                logger.info(f"Waiting {cycle_interval} seconds before next cycle...")
                time.sleep(cycle_interval)
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        income_report = self.income_manager.get_income_report()
        
        return {
            "income": income_report,
            "applications": {
                "total": len(self.job_memory.get("applications", [])),
                "recent": len([a for a in self.job_memory.get("applications", []) 
                              if datetime.fromisoformat(a.get("date", "2000-01-01")) > 
                              datetime.now().replace(hour=0, minute=0, second=0)])
            },
            "jobs_found": len(self.job_memory.get("jobs_found", [])),
            "status": "active"
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("GETAJOB - Autonomous Job-Seeking Agent System")
    logger.info("="*60)
    
    # Initialize orchestrator
    orchestrator = JobSeekerOrchestrator()
    
    # Get initial status
    status = orchestrator.get_status_report()
    logger.info(f"Initial status: {status}")
    
    # Run continuous job search
    # Adjust parameters as needed:
    # - max_cycles: None for infinite, or a number for limited cycles
    # - cycle_interval: seconds between cycles (3600 = 1 hour)
    
    try:
        orchestrator.run_continuous(
            max_cycles=None,  # Run indefinitely
            cycle_interval=3600  # 1 hour between cycles
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
        final_status = orchestrator.get_status_report()
        logger.info(f"Final status: {final_status}")
        
        # Final income report
        final_report = orchestrator.income_manager.report_to_corporation()
        logger.info(final_report)

if __name__ == "__main__":
    main()

