"""
Evaluation utilities for model comparison and accuracy testing.
"""
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

class ModelEvaluator:
    """Class to handle model evaluation and benchmarking."""
    
    def __init__(self):
        self.test_results = []
        # Updated benchmark queries for log analysis
        self.benchmark_queries = [
            "What errors occurred in the logs?",
            "Show me all warning messages",
            "What time did the system start?",
            "Find any connection timeouts",
            "List all successful operations",
            "What's the most common error type?",
            "Show authentication failures",
            "Find performance issues",
            "What IP addresses appear most frequently?",
            "Show all database connection errors"
        ]
    
    def create_test_suite(self, custom_queries: List[str] = None) -> List[str]:
        """Create a comprehensive test suite."""
        if custom_queries:
            return self.benchmark_queries + custom_queries
        return self.benchmark_queries
    
    def evaluate_response_quality(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """Evaluate response quality based on various metrics."""
        metrics = {
            "response_length": len(response),
            "context_usage": self._calculate_context_usage(response, context),
            "specificity_score": self._calculate_specificity(response),
            "contains_fallback": "I couldn't find this in the logs" in response,
            "timestamp": datetime.now().isoformat()
        }
        return metrics
    
    def _calculate_context_usage(self, response: str, context: str) -> float:
        """Calculate how much of the context was used in the response."""
        if not context or not response:
            return 0.0
        
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        
        if len(context_words) == 0:
            return 0.0
        
        overlap = len(context_words.intersection(response_words))
        return overlap / len(context_words)
    
    def _calculate_specificity(self, response: str) -> float:
        """Calculate specificity score based on response characteristics."""
        # Simple heuristic: longer responses with specific terms are more specific
        specific_terms = [
            "error", "warning", "timestamp", "ip", "port", "status", 
            "code", "message", "failed", "success", "timeout"
        ]
        
        response_lower = response.lower()
        specific_count = sum(1 for term in specific_terms if term in response_lower)
        
        # Normalize by response length and specific terms found
        if len(response) == 0:
            return 0.0
        
        return min(1.0, (specific_count * 0.2) + (len(response) / 1000 * 0.1))
    
    def run_comprehensive_evaluation(self, chat_states: Dict[str, Any], 
                                   test_queries: List[str] = None) -> pd.DataFrame:
        """Run comprehensive evaluation across all models."""
        if test_queries is None:
            test_queries = self.benchmark_queries
        
        results = []
        
        for query in test_queries:
            print(f"Testing query: {query}")
            
            for model_name, chat_state in chat_states.items():
                try:
                    start_time = time.time()
                    
                    # Get response (assuming run_chat function is available)
                    from .chat import run_chat
                    response, _, performance_metrics = run_chat(chat_state, query)
                    
                    # Get context for quality evaluation
                    retrieved_docs = chat_state["retriever"].get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # Evaluate response quality
                    quality_metrics = self.evaluate_response_quality(query, response, context)
                    
                    # Combine all metrics
                    result = {
                        "query": query,
                        "model_name": model_name,
                        "response": response,
                        "success": True,
                        **performance_metrics,
                        **quality_metrics
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    result = {
                        "query": query,
                        "model_name": model_name,
                        "response": f"Error: {str(e)}",
                        "success": False,
                        "error": str(e)
                    }
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if results_df.empty:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful_results = results_df[results_df['success'] == True]
        
        if successful_results.empty:
            return {"error": "No successful results to analyze"}
        
        # Calculate aggregate metrics per model
        model_stats = successful_results.groupby('model_name').agg({
            'total_time': ['mean', 'std', 'min', 'max'],
            'llm_time': ['mean', 'std'],
            'retrieval_time': ['mean', 'std'],
            'response_length': ['mean', 'std'],
            'context_usage': ['mean', 'std'],
            'specificity_score': ['mean', 'std'],
            'contains_fallback': 'sum'
        }).round(3)
        
        # Flatten column names
        model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns]
        
        # Calculate rankings
        rankings = {
            'fastest_avg_response': model_stats['total_time_mean'].idxmin(),
            'most_consistent': model_stats['total_time_std'].idxmin(),
            'best_context_usage': model_stats['context_usage_mean'].idxmax(),
            'most_specific': model_stats['specificity_score_mean'].idxmax(),
            'least_fallbacks': model_stats['contains_fallback_sum'].idxmin()
        }
        
        # Query-level analysis
        query_performance = successful_results.groupby('query').agg({
            'total_time': 'mean',
            'context_usage': 'mean',
            'specificity_score': 'mean'
        }).round(3)
        
        report = {
            "evaluation_summary": {
                "total_queries_tested": len(results_df['query'].unique()),
                "total_models_tested": len(results_df['model_name'].unique()),
                "overall_success_rate": len(successful_results) / len(results_df),
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "model_statistics": model_stats.to_dict(),
            "rankings": rankings,
            "query_performance": query_performance.to_dict(),
            "recommendations": self._generate_recommendations(model_stats, rankings)
        }
        
        return report
    
    def _generate_recommendations(self, model_stats: pd.DataFrame, 
                                rankings: Dict[str, str]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Speed recommendations
        fastest_model = rankings['fastest_avg_response']
        recommendations.append(
            f"For fastest response time, use {fastest_model}"
        )
        
        # Consistency recommendations
        most_consistent = rankings['most_consistent']
        recommendations.append(
            f"For most consistent performance, use {most_consistent}"
        )
        
        # Quality recommendations
        best_context = rankings['best_context_usage']
        recommendations.append(
            f"For best context utilization, use {best_context}"
        )
        
        most_specific = rankings['most_specific']
        recommendations.append(
            f"For most specific responses, use {most_specific}"
        )
        
        # Reliability recommendations
        least_fallbacks = rankings['least_fallbacks']
        recommendations.append(
            f"For most reliable responses (fewer fallbacks), use {least_fallbacks}"
        )
        
        return recommendations

def save_evaluation_results(results_df: pd.DataFrame, report: Dict[str, Any], 
                          filename_prefix: str = "model_evaluation"):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_filename = f"{filename_prefix}_results_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    
    # Save report
    report_filename = f"{filename_prefix}_report_{timestamp}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    return results_filename, report_filename
