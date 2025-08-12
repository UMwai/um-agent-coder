#!/usr/bin/env python3
"""
Main entry point for the Data-focused Agent.
Specialized for data engineering, data science, and ML tasks.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from um_agent_coder.config import Config
from um_agent_coder.llm.factory import LLMFactory
from um_agent_coder.agent.data_agent import DataAgent
from um_agent_coder.agent.data_modes import DataAgentMode


def main():
    """Main entry point for the data agent."""
    parser = argparse.ArgumentParser(
        description='UM Data Agent - Specialized for data engineering and science tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Data Engineering
  um-data-agent "Create an ETL pipeline to process daily sales data from PostgreSQL to Snowflake"
  um-data-agent "Design a star schema for an e-commerce data warehouse" --mode data_architect
  
  # Data Science
  um-data-agent "Analyze customer churn patterns in subscription_data.csv" --data subscription_data.csv
  um-data-agent "Build a predictive model for sales forecasting" --mode data_scientist
  
  # ML Engineering
  um-data-agent "Deploy the trained model as a REST API with monitoring" --mode ml_engineer
  um-data-agent "Set up a feature store for real-time ML features" --mode ml_engineer
  
  # Analytics Engineering
  um-data-agent "Create dbt models for financial reporting" --mode analytics_engineer --output dbt
  um-data-agent "Build a metrics layer for product analytics" --mode analytics_engineer

Available Modes:
  - data_engineer: ETL/ELT pipelines, data infrastructure
  - data_architect: Data modeling, schema design, governance
  - data_scientist: ML models, statistical analysis, EDA
  - ml_engineer: MLOps, model deployment, monitoring
  - analytics_engineer: dbt, transformations, metrics
  - data_analyst: Analysis, reporting, visualization
        """
    )
    
    parser.add_argument('prompt', help='The data task to execute')
    parser.add_argument(
        '--mode', '-m',
        choices=['data_engineer', 'data_architect', 'data_scientist', 
                 'ml_engineer', 'analytics_engineer', 'data_analyst'],
        help='Specific data mode to use (auto-detected if not specified)'
    )
    parser.add_argument(
        '--data', '-d',
        help='Data source (file path or connection string)'
    )
    parser.add_argument(
        '--output', '-o',
        choices=['sql', 'python', 'dbt', 'airflow', 'prefect', 'yaml', 'json'],
        help='Output format for generated artifacts'
    )
    parser.add_argument(
        '--model',
        help='Specific LLM model to use (e.g., claude-3.5-sonnet, gpt-4o)'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-profile',
        action='store_true',
        help='Disable automatic data profiling'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Disable automatic data validation'
    )
    parser.add_argument(
        '--export-lineage',
        help='Export data lineage to file after execution'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Override config with command line arguments
        if args.verbose:
            config.set('verbose', True)
        
        if args.no_profile:
            config.set('auto_profile', False)
        
        if args.no_validate:
            config.set('validate_data', False)
        
        # Select model
        if args.model:
            model_name = args.model
        else:
            # Auto-select based on mode
            mode_map = {
                'data_engineer': 'claude-3.5-sonnet-20241022',
                'data_architect': 'claude-3.5-sonnet-20241022',
                'data_scientist': 'gpt-4o',
                'ml_engineer': 'claude-3.5-sonnet-20241022',
                'analytics_engineer': 'claude-3.5-sonnet-20241022',
                'data_analyst': 'gemini-2.0-flash'  # Cost-effective for queries
            }
            model_name = mode_map.get(args.mode, 'claude-3.5-sonnet-20241022')
        
        config.set('llm.model', model_name)
        
        # Create LLM instance
        llm = LLMFactory.create(config)
        
        # Create Data Agent
        agent = DataAgent(llm, config.data)
        
        # Convert mode string to enum
        mode = None
        if args.mode:
            mode = DataAgentMode(args.mode)
        
        # Execute the task
        print(f"\n🚀 Starting Data Agent with model: {model_name}")
        if args.mode:
            print(f"📊 Mode: {args.mode}")
        if args.data:
            print(f"📁 Data source: {args.data}")
        print(f"💭 Task: {args.prompt}\n")
        
        result = agent.run(
            prompt=args.prompt,
            mode=mode,
            data_source=args.data,
            output_format=args.output
        )
        
        # Display results
        print("\n" + "="*60)
        print("📊 RESULTS")
        print("="*60)
        
        if result['success']:
            print(f"\n✅ Task completed successfully!")
            print(f"\n{result['response']}")
            
            # Show artifacts if any
            if result.get('artifacts'):
                print("\n" + "-"*60)
                print("📦 Generated Artifacts:")
                for name, content in result['artifacts'].items():
                    print(f"\n• {name}:")
                    if isinstance(content, dict):
                        if 'files_created' in content:
                            for file in content['files_created']:
                                print(f"  - {file}")
                    else:
                        print(f"  {str(content)[:200]}...")
            
            # Show data insights
            if result.get('data_insights'):
                insights = result['data_insights']
                if insights.get('data_quality_issues'):
                    print("\n" + "-"*60)
                    print("⚠️ Data Quality Issues:")
                    for issue in insights['data_quality_issues'][:5]:
                        print(f"  • {issue}")
                
                if insights.get('statistics'):
                    print("\n" + "-"*60)
                    print("📈 Data Statistics:")
                    for key, value in insights['statistics'].items():
                        print(f"  • {key}: {value}")
            
            # Show execution details
            if result.get('execution_details'):
                details = result['execution_details']
                print("\n" + "-"*60)
                print("⚙️ Execution Details:")
                print(f"  • Steps executed: {details['steps_executed']}")
                print(f"  • Tools used: {', '.join(details['tools_used'])}")
                if details.get('data_processed'):
                    volume = details['data_processed']
                    if volume.get('rows_processed') > 0:
                        print(f"  • Rows processed: {volume['rows_processed']:,}")
                    if volume.get('tables_analyzed') > 0:
                        print(f"  • Tables analyzed: {volume['tables_analyzed']}")
            
            # Export lineage if requested
            if args.export_lineage and result.get('lineage'):
                lineage_path = Path(args.export_lineage)
                task_id = result['task_id']
                
                if lineage_path.suffix == '.json':
                    lineage_content = agent.export_pipeline(task_id, 'json')
                else:
                    lineage_content = agent.export_pipeline(task_id, 'yaml')
                
                lineage_path.write_text(lineage_content)
                print(f"\n💾 Data lineage exported to: {lineage_path}")
            
            # Show cost metrics
            if result.get('metrics'):
                metrics = result['metrics']
                if metrics.get('total_cost', 0) > 0:
                    print("\n" + "-"*60)
                    print("💰 Cost Metrics:")
                    print(f"  • Total cost: ${metrics['total_cost']:.4f}")
                    print(f"  • Total tokens: {metrics.get('total_tokens', 0):,}")
        else:
            print(f"\n❌ Task failed: {result.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())