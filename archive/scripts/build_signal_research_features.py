import time

from final_multi_asset_project_common import ensure_research_inputs


def main() -> None:
    start_time = time.perf_counter()
    outputs = ensure_research_inputs()
    elapsed = time.perf_counter() - start_time
    for stem, paths in outputs.items():
        print(f"{stem}: raw={paths['raw_path']}")
        print(f"{stem}: feature={paths['feature_path']}")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
