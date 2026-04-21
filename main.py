from src.preprocessing import load_data
from src.analysis import run_analysis
from src.model import run_model
from src.strategy import run_strategy
from src.insights import generate_insights

def main():
    data = load_data()

    analysis_results = run_analysis(data)
    model_results = run_model(data)
    strategy_results = run_strategy(data)

    generate_insights(analysis_results, model_results, strategy_results)
    print("\nAll outputs saved inside the outputs/ folder.")

if __name__ == "__main__":
    main()
