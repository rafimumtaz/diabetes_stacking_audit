from .config import OUTPUT_DIR
from .data_utils import load_dataset, select_three_classes
from .audit import run_dataset_audit
from .modeling import run_model_experiments

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUDIT DATASET DIABETES DAN EKSPERIMEN ABLATION")
    print("=" * 70)

    full_df = load_dataset()
    print(f"Dataset awal: {full_df.shape[0]} baris x {full_df.shape[1]} kolom")

    selected_df = select_three_classes(full_df)
    print(f"Subset tiga kelas: {selected_df.shape[0]} baris x {selected_df.shape[1]} kolom")
    print("\nDistribusi kelas:")
    print(selected_df["Target"].value_counts())

    run_dataset_audit(full_df, selected_df)
    run_model_experiments(selected_df)

    print("\nSemua proses selesai.")
    print(f"Periksa hasil di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
