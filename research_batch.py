#!/usr/bin/env python3
"""
Batch Research Script for Critical Forest Height Estimation Topics
Total: 15 research rounds (5 rounds per topic × 3 critical topics)
"""

import subprocess
import time
import os
from datetime import datetime

# Critical Topics with 5 research queries each
research_topics = {
    "topic1_stereo_matching": [
        "semi-global matching SGM PatchMatch stereo photogrammetry 2025",
        "deep learning stereo matching PSM-Net RAFT-Stereo 2024 2025",
        "multi-view stereo forest canopy dense matching 2025",
        "uncertainty-aware stereo matching satellite imagery 2024",
        "real-time stereo matching GPU optimization 2025"
    ],
    "topic2_multi_sensor_fusion": [
        "cross-attention fusion stereo LiDAR SAR forest height 2024 2025",
        "transformer-based multi-sensor fusion remote sensing 2025",
        "hierarchical fusion deep learning satellite airborne LiDAR",
        "bayesian uncertainty quantification multi-sensor fusion 2024",
        "multi-modal feature fusion architecture forest analysis 2025"
    ],
    "topic3_deep_learning_models": [
        "U-Net canopy height model training dataset 2024 2025",
        "transformer vision models forest height estimation 2025",
        "foundation models depth estimation forestry adaptation",
        "multi-task learning height biomass cover 2024",
        "attention mechanisms CNN forest structure 2024 2025"
    ]
}

def run_research(query, topic_name, round_num):
    """Execute a single research round"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    print(f"\n{'='*80}")
    print(f"🚀 [{timestamp}] Starting {topic_name} - Round {round_num}/5")
    print(f"📝 Query: {query}")
    print(f"{'='*80}\n")
    
    try:
        # Run research agent
        result = subprocess.run(
            ['python3', 'main.py', query],
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes per query
        )
        
        print(f"✅ {topic_name} - Round {round_num} Complete")
        print(f"   Report saved to reports/")
        
        # Wait between queries to avoid rate limiting
        time.sleep(10)
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  {topic_name} - Round {round_num} Timed out")
        return False
    except Exception as e:
        print(f"❌ {topic_name} - Round {round_num} Failed: {e}")
        return False

def main():
    """Execute all research rounds"""
    print("\n" + "="*80)
    print("🚀 BATCH RESEARCH - CRITICAL FOREST HEIGHT ESTIMATION TOPICS")
    print("="*80)
    print(f"Total Topics: {len(research_topics)}")
    print(f"Total Research Rounds: {sum(len(queries) for queries in research_topics.values())}")
    print(f"Estimated Time: 2-3 hours")
    print("="*80 + "\n")
    
    results = {
        'success': 0,
        'failed': 0,
        'timed_out': 0
    }
    
    # Execute research rounds
    for topic_name, queries in research_topics.items():
        print(f"\n{'#'*80}")
        print(f"# TOPIC: {topic_name.upper()}")
        print(f"{'#'*80}\n")
        
        for round_num, query in enumerate(queries, 1):
            success = run_research(query, topic_name, round_num)
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
            
            # Progress summary
            completed = results['success'] + results['failed']
            total = sum(len(q) for q in research_topics.values())
            print(f"\n📊 Progress: {completed}/{total} rounds completed")
            print(f"   Success: {results['success']}, Failed: {results['failed']}")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 RESEARCH BATCH SUMMARY")
    print("="*80)
    print(f"Total Rounds: {sum(len(q) for q in research_topics.values())}")
    print(f"✅ Successful: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⏱️  Timed Out: {results['timed_out']}")
    print(f"Success Rate: {results['success']/sum(len(q) for q in research_topics.values())*100:.1f}%")
    print("="*80 + "\n")
    
    # List generated reports
    print("📁 Generated Reports:")
    print("="*80)
    os.system('ls -lht reports/*.md | head -20')
    print("="*80 + "\n")

if __name__ == "__main__":
    main()