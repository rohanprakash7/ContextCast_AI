"""
Generate Professional Comparison Report
Creates both Markdown and HTML reports from comparison results
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def load_comparison_results(json_file: str) -> dict:
    """Load comparison results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)


def generate_markdown_report(comparison: dict, output_file: str = None):
    """Generate comprehensive Markdown report"""
    
    if output_file is None:
        output_file = f"RAG_Evaluation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    report = []
    
    # Header
    report.append("# RAG System Evaluation Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Video:** {comparison['baseline']['video_id']}")
    report.append(f"\n**Questions Evaluated:** {comparison['baseline']['num_questions']}")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    best = comparison['best_config']
    report.append(f"**Best Configuration:** `{best['config_name']}`\n")
    report.append(f"- **Overall Score:** {best['overall_score']:.3f}")
    report.append(f"- **Faithfulness:** {best['faithfulness']:.3f} (Perfect = 1.0)")
    report.append(f"- **Context Precision:** {best['context_precision']:.3f}")
    report.append(f"- **Context Recall:** {best['context_recall']:.3f}\n")
    
    # Key Findings
    report.append("### Key Findings\n")
    for rec in comparison['recommendations']:
        report.append(f"- {rec}")
    report.append("\n---\n")
    
    # Detailed Results
    report.append("## Detailed Results\n")
    report.append("### All Configurations Tested\n")
    report.append("| Configuration | Faithfulness | Precision | Recall | Overall Score |")
    report.append("|--------------|--------------|-----------|--------|---------------|")
    
    for result in comparison['all_results']:
        if 'error' not in result:
            # Calculate overall score
            score = (result['faithfulness'] * 0.4 + 
                    result['context_recall'] * 0.4 + 
                    result['context_precision'] * 0.2)
            report.append(f"| {result['config_name']:<12} | "
                         f"{result['faithfulness']:.3f} | "
                         f"{result['context_precision']:.3f} | "
                         f"{result['context_recall']:.3f} | "
                         f"{score:.3f} |")
    
    report.append("\n")
    
    # Improvements Over Baseline
    report.append("### Improvements Over Baseline\n")
    baseline = comparison['baseline']
    report.append(f"**Baseline Configuration:** `{baseline['config_name']}`\n")
    report.append("| Configuration | Faithfulness | Precision | Recall |")
    report.append("|--------------|--------------|-----------|--------|")
    
    for improvement in comparison['improvements']:
        f_change = improvement['faithfulness_change']
        p_change = improvement['precision_change']
        r_change = improvement['recall_change']
        
        report.append(f"| {improvement['config']:<12} | "
                     f"{f_change:+.3f} | "
                     f"{p_change:+.3f} | "
                     f"{r_change:+.3f} |")
    
    report.append("\n---\n")
    
    # Metric Explanations
    report.append("## Metric Explanations\n")
    report.append("### Faithfulness (0.0 - 1.0)")
    report.append("Measures if answers are grounded in retrieved context without hallucination.")
    report.append("- **1.0:** Perfect - No hallucinations")
    report.append("- **0.8-1.0:** Excellent")
    report.append("- **< 0.8:** Needs improvement\n")
    
    report.append("### Context Precision (0.0 - 1.0)")
    report.append("Measures relevance of retrieved chunks to the question.")
    report.append("- **> 0.9:** Excellent - Retrieving very relevant chunks")
    report.append("- **0.7-0.9:** Good")
    report.append("- **< 0.7:** Needs better retrieval\n")
    
    report.append("### Context Recall (0.0 - 1.0)")
    report.append("Measures completeness - did we retrieve all relevant information?")
    report.append("- **> 0.8:** Excellent - Very complete retrieval")
    report.append("- **0.6-0.8:** Good")
    report.append("- **< 0.6:** Missing important information\n")
    
    report.append("---\n")
    
    # Recommendations
    report.append("## Recommendations\n")
    report.append(f"### Recommended Configuration: `{best['config_name']}`\n")
    report.append("**Why this configuration?**\n")
    report.append(f"- Achieves best balance of all metrics (Overall Score: {best['overall_score']:.3f})")
    report.append(f"- Perfect faithfulness ({best['faithfulness']:.3f}) - No hallucinations")
    report.append(f"- Strong recall ({best['context_recall']:.3f}) - Retrieves most relevant information")
    report.append(f"- Good precision ({best['context_precision']:.3f}) - Retrieved chunks are relevant\n")
    
    report.append("### Implementation Steps\n")
    report.append("```python")
    report.append("# Update your RAG system to use the best configuration:")
    report.append("result = answer_question(")
    report.append("    question=question,")
    report.append("    video_id=video_id,")
    report.append(f"    retriever_type='{best['retriever_type']}',")
    report.append(f"    top_k={best['top_k']}")
    report.append(")")
    report.append("```\n")
    
    # Configuration Details
    report.append("---\n")
    report.append("## Configuration Details\n")
    
    for i, result in enumerate(comparison['all_results'], 1):
        if 'error' not in result:
            report.append(f"### {i}. {result['config_name']}\n")
            report.append(f"- **Retriever Type:** {result['retriever_type']}")
            report.append(f"- **Top K:** {result['top_k']}")
            report.append(f"- **Faithfulness:** {result['faithfulness']:.3f}")
            report.append(f"- **Context Precision:** {result['context_precision']:.3f}")
            report.append(f"- **Context Recall:** {result['context_recall']:.3f}")
            
            # Add interpretation
            if result['faithfulness'] >= 0.95:
                report.append(f"- ✅ Excellent faithfulness - minimal hallucination")
            if result['context_recall'] >= 0.8:
                report.append(f"- ✅ Excellent recall - retrieving complete information")
            elif result['context_recall'] >= 0.6:
                report.append(f"- ⚠️ Good recall but could be improved")
            else:
                report.append(f"- ❌ Low recall - missing important information")
            
            report.append("\n")
    
    # Footer
    report.append("---\n")
    report.append(f"\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ Markdown report saved to: {output_file}")
    return output_file


def generate_html_report(comparison: dict, output_file: str = None):
    """Generate HTML report with styling"""
    
    if output_file is None:
        output_file = f"RAG_Evaluation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    html = []
    
    # HTML Header with CSS
    html.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Evaluation Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }
        h3 {
            color: #7f8c8d;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: #27ae60;
            font-weight: bold;
        }
        .negative {
            color: #e74c3c;
            font-weight: bold;
        }
        .recommendation {
            background-color: #e8f8f5;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
        }
        .code-block {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .badge-excellent {
            background-color: #27ae60;
            color: white;
        }
        .badge-good {
            background-color: #f39c12;
            color: white;
        }
        .badge-poor {
            background-color: #e74c3c;
            color: white;
        }
        .timestamp {
            color: #95a5a6;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
""")
    
    # Title and Header
    html.append(f"""
        <h1>RAG System Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Video ID:</strong> {comparison['baseline']['video_id']}</p>
        <p><strong>Questions Evaluated:</strong> {comparison['baseline']['num_questions']}</p>
    """)
    
    # Best Configuration Card
    best = comparison['best_config']
    html.append(f"""
        <div class="metric-card">
            <h2 style="margin-top: 0; color: white; border: none;">🏆 Best Configuration</h2>
            <h3 style="color: white;"><code>{best['config_name']}</code></h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 20px;">
                <div>
                    <div style="opacity: 0.8;">Overall Score</div>
                    <div class="metric-value">{best['overall_score']:.3f}</div>
                </div>
                <div>
                    <div style="opacity: 0.8;">Faithfulness</div>
                    <div class="metric-value">{best['faithfulness']:.3f}</div>
                </div>
                <div>
                    <div style="opacity: 0.8;">Precision</div>
                    <div class="metric-value">{best['context_precision']:.3f}</div>
                </div>
                <div>
                    <div style="opacity: 0.8;">Recall</div>
                    <div class="metric-value">{best['context_recall']:.3f}</div>
                </div>
            </div>
        </div>
    """)
    
    # Recommendations
    html.append("<h2>📋 Key Recommendations</h2>")
    for rec in comparison['recommendations']:
        html.append(f'<div class="recommendation">{rec}</div>')
    
    # Comparison Table
    html.append("<h2>📊 All Configurations Compared</h2>")
    html.append("<table>")
    html.append("<tr><th>Configuration</th><th>Faithfulness</th><th>Precision</th><th>Recall</th><th>Overall</th></tr>")
    
    for result in comparison['all_results']:
        if 'error' not in result:
            score = (result['faithfulness'] * 0.4 + 
                    result['context_recall'] * 0.4 + 
                    result['context_precision'] * 0.2)
            
            # Add badges
            f_badge = "excellent" if result['faithfulness'] >= 0.95 else "good" if result['faithfulness'] >= 0.8 else "poor"
            p_badge = "excellent" if result['context_precision'] >= 0.9 else "good" if result['context_precision'] >= 0.7 else "poor"
            r_badge = "excellent" if result['context_recall'] >= 0.8 else "good" if result['context_recall'] >= 0.6 else "poor"
            
            html.append(f"""
                <tr>
                    <td><strong>{result['config_name']}</strong></td>
                    <td><span class="badge badge-{f_badge}">{result['faithfulness']:.3f}</span></td>
                    <td><span class="badge badge-{p_badge}">{result['context_precision']:.3f}</span></td>
                    <td><span class="badge badge-{r_badge}">{result['context_recall']:.3f}</span></td>
                    <td><strong>{score:.3f}</strong></td>
                </tr>
            """)
    
    html.append("</table>")
    
    # Improvements Table
    html.append("<h2>📈 Improvements Over Baseline</h2>")
    baseline = comparison['baseline']
    html.append(f"<p><strong>Baseline:</strong> {baseline['config_name']}</p>")
    html.append("<table>")
    html.append("<tr><th>Configuration</th><th>Faithfulness</th><th>Precision</th><th>Recall</th></tr>")
    
    for improvement in comparison['improvements']:
        f_change = improvement['faithfulness_change']
        p_change = improvement['precision_change']
        r_change = improvement['recall_change']
        
        f_class = "positive" if f_change > 0 else "negative" if f_change < 0 else ""
        p_class = "positive" if p_change > 0 else "negative" if p_change < 0 else ""
        r_class = "positive" if r_change > 0 else "negative" if r_change < 0 else ""
        
        html.append(f"""
            <tr>
                <td><strong>{improvement['config']}</strong></td>
                <td class="{f_class}">{f_change:+.3f}</td>
                <td class="{p_class}">{p_change:+.3f}</td>
                <td class="{r_class}">{r_change:+.3f}</td>
            </tr>
        """)
    
    html.append("</table>")
    
    # Implementation Code
    html.append("<h2>💻 Implementation</h2>")
    html.append(f"""
        <p>Update your RAG system to use the best configuration:</p>
        <div class="code-block">
result = answer_question(<br>
&nbsp;&nbsp;&nbsp;&nbsp;question=question,<br>
&nbsp;&nbsp;&nbsp;&nbsp;video_id=video_id,<br>
&nbsp;&nbsp;&nbsp;&nbsp;retriever_type='{best['retriever_type']}',<br>
&nbsp;&nbsp;&nbsp;&nbsp;top_k={best['top_k']}<br>
)
        </div>
    """)
    
    # Footer
    html.append(f"""
        <hr style="margin-top: 40px;">
        <p class="timestamp">Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
""")
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    
    print(f"✅ HTML report saved to: {output_file}")
    return output_file


def main():
    """Main function to generate reports"""
    
    # Find the most recent comparison JSON file
    json_files = list(Path('.').glob('evaluation_comparison_*.json'))
    
    if not json_files:
        print("❌ No comparison JSON files found!")
        print("Please run the comparison evaluator first: python evaluation/comparison_evaluator.py")
        return
    
    # Use the most recent file
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    print(f"\n📁 Loading comparison results from: {latest_file}")
    
    # Load results
    comparison = load_comparison_results(str(latest_file))
    
    # Generate both reports
    print("\n📝 Generating reports...")
    md_file = generate_markdown_report(comparison)
    html_file = generate_html_report(comparison)
    
    print("\n" + "="*80)
    print("✅ REPORTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📄 Markdown: {md_file}")
    print(f"   - Easy to read in any text editor")
    print(f"   - Can be committed to Git/GitHub")
    print(f"   - Good for documentation")
    
    print(f"\n🌐 HTML: {html_file}")
    print(f"   - Professional formatting")
    print(f"   - Open in any web browser")
    print(f"   - Great for presentations")
    
    print("\n💡 TIP: Open the HTML file in your browser for the best viewing experience!")
    print("="*80)


if __name__ == "__main__":
    main()