from build_signal_research_features import main as build_signal_research_features_main
from run_signal_research_models import main as run_signal_research_models_main
from run_trade_execution_backtrader import main as run_trade_execution_backtrader_main


def main() -> None:
    build_signal_research_features_main()
    run_signal_research_models_main()
    run_trade_execution_backtrader_main()


if __name__ == "__main__":
    main()
