import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV data
try:
    # Load the CSV file directly as text
    file_path = 'data/local/vpg-maml-point100/trpomaml1_fbs20_mbs40_flr_0.5metalr_0.01_step11/progress.csv'
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    print("Successfully loaded CSV file!")
    
    # Print column info
    print(f"Number of iterations: {len(df)}")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot the learning curves - average returns
    iterations = range(len(df))
    ax1.plot(iterations, df['0AverageReturn'], label='Pre-Adaptation', color='blue')
    ax1.plot(iterations, df['1AverageReturn'], label='Post-Adaptation (1 gradient step)', color='green')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Return')
    ax1.set_title('MAML Learning Curves for Point Navigation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot the improvement ratio over time
    improvement_ratio = np.abs(df['0AverageReturn'] / df['1AverageReturn'])
    ax2.plot(iterations, improvement_ratio, label='Improvement Factor', color='red')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Improvement Factor (Pre/Post)')
    ax2.set_title('MAML Adaptation Improvement Factor')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('maml_point_results.png')
    print("Saved plot to maml_point_results.png")
    
    # Print summary statistics
    print("\nSummary of training results:")
    print(f"Initial iteration:")
    print(f"  Pre-adaptation return: {df['0AverageReturn'].iloc[0]:.2f}")
    print(f"  Post-adaptation return: {df['1AverageReturn'].iloc[0]:.2f}")
    print(f"  Improvement factor: {abs(df['0AverageReturn'].iloc[0]/df['1AverageReturn'].iloc[0]):.2f}x")
    
    print(f"\nFinal iteration:")
    print(f"  Pre-adaptation return: {df['0AverageReturn'].iloc[-1]:.2f}")
    print(f"  Post-adaptation return: {df['1AverageReturn'].iloc[-1]:.2f}")
    print(f"  Improvement factor: {abs(df['0AverageReturn'].iloc[-1]/df['1AverageReturn'].iloc[-1]):.2f}x")
    
    # Calculate improvement over training
    initial_pre = df['0AverageReturn'].iloc[0]
    initial_post = df['1AverageReturn'].iloc[0]
    final_pre = df['0AverageReturn'].iloc[-1]
    final_post = df['1AverageReturn'].iloc[-1]
    
    print(f"\nImprovement over training:")
    print(f"  Pre-adaptation: {(final_pre - initial_pre)/abs(initial_pre)*100:.1f}% improvement")
    print(f"  Post-adaptation: {(final_post - initial_post)/abs(initial_post)*100:.1f}% improvement")
    
except Exception as e:
    print(f"Error: {e}")