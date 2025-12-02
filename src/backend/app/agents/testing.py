from typing import Dict, Any, List, Optional
import sys
import json
from pathlib import Path
from datetime import datetime
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from app.agents.base import BaseAgent
from app.models.agent_metadata import IntentType
from app.core.agent_registry import AgentRegistry
from app.utils.logger import setup_logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = setup_logger("testing_agent")


class TestingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="testing_agent",
            description="Tests knowledge agent responses against structured test suite",
            intents=[IntentType.SYSTEM_TESTING],
            capabilities=["test_execution", "response_validation", "comparison"],
            priority=1,
            requires_user_id=False
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.TESTING_TEMPERATURE
        )
        self.registry = AgentRegistry()
        self.test_suite = self._load_test_suite()
    
    def _load_test_suite(self) -> List[Dict[str, Any]]:
        """Load test suite from JSON file in utils directory"""
        try:
            current_file = Path(__file__).resolve()
            utils_dir = current_file.parent.parent / 'utils'
            test_file = utils_dir / 'test_suite.json'
            
            if not test_file.exists():
                logger.warning(f"Test suite file not found at {test_file}. Creating empty test suite.")
                return []
            
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_cases = data.get('test_cases', [])
                logger.info(f"Loaded {len(test_cases)} test cases from test suite")
                return test_cases
        except Exception as e:
            logger.error(f"Error loading test suite: {str(e)}")
            return []
    
    def _find_test_case(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a test case by question or return None for custom queries"""
        query_lower = query.lower().strip()
        
        for test_case in self.test_suite:
            if test_case['question'].lower().strip() == query_lower:
                return test_case
        
        return None
    
    async def _get_agent_response(self, query: str) -> Dict[str, Any]:
        """Get response from knowledge agent"""
        try:
            knowledge_agent_metadata = self.registry.get_agent("knowledge_agent")
            if not knowledge_agent_metadata or not knowledge_agent_metadata.agent_instance:
                logger.warning("Knowledge agent not found in registry")
                return {
                    "response": "Knowledge agent not available",
                    "agent": "unknown",
                    "metadata": {}
                }
            
            knowledge_agent = knowledge_agent_metadata.agent_instance
            context = {
                "message": query,
                "user_id": None,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Getting response from knowledge agent for: '{query[:100]}...'")
            response = await knowledge_agent.process(context)
            return response
        except Exception as e:
            logger.error(f"Error getting agent response: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "agent": "knowledge_agent",
                "metadata": {"error": str(e)}
            }
    
    async def _compare_responses(
        self,
        actual_response: str,
        expected_answer: str,
        question: str,
        source_url: str
    ) -> Dict[str, Any]:
        """Compare actual agent response with expected answer using LLM"""
        prompt = f"""You are a validation system that compares an actual response from a knowledge agent with an expected answer.

Question: "{question}"
Source URL: {source_url}

Expected Answer (what the response should contain):
{expected_answer}

Actual Response (from the knowledge agent):
{actual_response}

Analyze and compare these responses. Determine:
1. Does the actual response contain the key information from the expected answer? (match: true/false)
2. What is the confidence level? (0.0 to 1.0)
3. What are the key differences, if any?
4. What information matches between them?
5. Is the actual response accurate and complete?

Note: The actual response may be more detailed or phrased differently, but should contain the core information from the expected answer.

Respond in this format:
match: <true or false>
confidence: <0.0 to 1.0>
differences: <list of key differences, or "none" if they match>
similarities: <list of matching information>
reason: <brief explanation>"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            result = {
                "match": False,
                "confidence": 0.0,
                "reason": "Could not parse comparison",
                "differences": [],
                "similarities": []
            }
            
            for line in response_text.split('\n'):
                line_lower = line.lower().strip()
                if 'match:' in line_lower:
                    result["match"] = 'true' in line_lower
                elif 'confidence:' in line_lower:
                    try:
                        conf_str = line.split('confidence:')[1].strip()
                        result["confidence"] = float(conf_str.split()[0])
                    except:
                        pass
                elif 'differences:' in line_lower:
                    diff_text = line.split('differences:')[1].strip()
                    if diff_text.lower() != 'none':
                        result["differences"] = [d.strip() for d in diff_text.split(',') if d.strip()]
                elif 'similarities:' in line_lower:
                    sim_text = line.split('similarities:')[1].strip()
                    result["similarities"] = [s.strip() for s in sim_text.split(',') if s.strip()]
                elif 'reason:' in line_lower:
                    result["reason"] = line.split('reason:')[1].strip()
            
            return result
        except Exception as e:
            logger.error(f"Error comparing responses: {str(e)}")
            return {
                "match": False,
                "confidence": 0.0,
                "reason": f"Comparison error: {str(e)}",
                "differences": [],
                "similarities": []
            }
    
    async def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        test_id = test_case.get('id', 'unknown')
        question = test_case.get('question', '')
        expected_answer = test_case.get('expected_answer', '')
        source_url = test_case.get('source_url', '')
        
        logger.info(f"Running test {test_id}: '{question[:100]}...'")
        
        start_time = datetime.now()
        
        try:
            agent_result = await self._get_agent_response(question)
            actual_response = agent_result.get("response", "")
            agent_metadata = agent_result.get("metadata", {})
            
            comparison = await self._compare_responses(
                actual_response,
                expected_answer,
                question,
                source_url
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            status = "PASS" if comparison["match"] and comparison["confidence"] > 0.7 else "FAIL"
            
            logger.info(
                f"Test {test_id} Result - Status: {status} | "
                f"Confidence: {comparison['confidence']:.2f} | "
                f"Processing Time: {int(processing_time)}ms"
            )
            
            return {
                "test_id": test_id,
                "question": question,
                "source_url": source_url,
                "status": status,
                "confidence": comparison["confidence"],
                "match": comparison["match"],
                "expected_answer": expected_answer,
                "actual_response": actual_response,
                "agent_metadata": agent_metadata,
                "comparison": comparison,
                "processing_time_ms": int(processing_time)
            }
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Test {test_id} Error: {str(e)}")
            return {
                "test_id": test_id,
                "question": question,
                "source_url": source_url,
                "status": "ERROR",
                "confidence": 0.0,
                "match": False,
                "error": str(e),
                "processing_time_ms": int(processing_time)
            }
    
    async def _run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases in the suite"""
        logger.info(f"Running all {len(self.test_suite)} test cases")
        
        start_time = datetime.now()
        results = []
        
        for test_case in self.test_suite:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        passed = sum(1 for r in results if r.get("status") == "PASS")
        failed = sum(1 for r in results if r.get("status") == "FAIL")
        errors = sum(1 for r in results if r.get("status") == "ERROR")
        
        logger.info(
            f"Test Suite Complete - Passed: {passed} | Failed: {failed} | Errors: {errors} | "
            f"Total Time: {int(total_time)}ms"
        )
        
        return {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / len(results) if results else 0.0,
            "total_time_ms": int(total_time),
            "results": results
        }
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process testing request"""
        message = context.get("message", "").strip()
        
        if not message:
            return {
                "response": "Please provide a test question or 'run_all' to run all tests.",
                "agent": self.name,
                "metadata": {
                    "status": "ERROR",
                    "error": "No message provided"
                }
            }
        
        if message.lower() == "run_all" or message.lower() == "run all":
            suite_results = await self._run_all_tests()
            
            summary = f"""Test Suite Results:

Total Tests: {suite_results['total_tests']}
Passed: {suite_results['passed']}
Failed: {suite_results['failed']}
Errors: {suite_results['errors']}
Pass Rate: {suite_results['pass_rate']:.1%}
Total Time: {suite_results['total_time_ms']}ms

Detailed Results:
"""
            for result in suite_results['results']:
                summary += f"\n{result['test_id']}: {result['status']} (Confidence: {result.get('confidence', 0):.2f})\n"
                summary += f"  Question: {result['question']}\n"
                if result.get('error'):
                    summary += f"  Error: {result['error']}\n"
            
            return {
                "response": summary,
                "agent": self.name,
                "metadata": suite_results
            }
        
        test_case = self._find_test_case(message)
        
        if test_case:
            result = await self._run_single_test(test_case)
            
            response_text = f"""Test Result:

Test ID: {result['test_id']}
Status: {result['status']}
Confidence: {result['confidence']:.2f}
Match: {'Yes' if result['match'] else 'No'}

Question: {result['question']}
Source URL: {result['source_url']}

Expected Answer:
{result['expected_answer']}

Actual Response:
{result['actual_response'][:500]}{'...' if len(result['actual_response']) > 500 else ''}

Comparison:
Reason: {result['comparison']['reason']}

Differences: {', '.join(result['comparison']['differences']) if result['comparison']['differences'] else 'None'}
Similarities: {', '.join(result['comparison']['similarities']) if result['comparison']['similarities'] else 'None'}

Processing Time: {result['processing_time_ms']}ms"""
            
            return {
                "response": response_text,
                "agent": self.name,
                "metadata": result
            }
        else:
            agent_result = await self._get_agent_response(message)
            actual_response = agent_result.get("response", "")
            
            return {
                "response": f"""Custom Query Test:

Question: {message}

Agent Response:
{actual_response[:500]}{'...' if len(actual_response) > 500 else ''}

Note: This is a custom query not in the test suite. No expected answer available for comparison.""",
                "agent": self.name,
                "metadata": {
                    "status": "CUSTOM_QUERY",
                    "question": message,
                    "actual_response": actual_response,
                    "agent_metadata": agent_result.get("metadata", {})
                }
            }
