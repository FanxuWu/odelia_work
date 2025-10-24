# F:\odelia_work\quick_check.py


from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from datasets import load_dataset

REPO_ID = "ODELIA-AI/ODELIA-Challenge-2025"
CONFIG = "unilateral"  # 或 "default"（视任务需要）

def main():
    api = HfApi()
    print("🔍 Checking login status...")
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("preferredUsername") or user_info.get("email")
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print("❌ You are not logged in. Please run:")
        print("   hf auth login")
        return

    print(f"\n🔍 Checking dataset access: {REPO_ID}")
    try:
        info = api.dataset_info(REPO_ID)
        print(f"✅ Dataset found!")
        print(f"   Private: {info.private}, Gated: {info.gated}, SHA: {info.sha[:7] if info.sha else 'n/a'}")
    except HfHubHTTPError as e:
        print(f"❌ Cannot access dataset metadata: {e}")
        print("👉 请确认：\n"
              "   1. 已在网页上点击 “Agree to Terms / Request Access”。\n"
              "   2. 你的 token 权限为 Read。\n"
              "   3. 网络连接正常（必要时使用 VPN）。")
        return

    print("\n🔍 Streaming a tiny sample from validation split (this may take a few seconds)...")
    try:
        ds = load_dataset(REPO_ID, name=CONFIG, split="val", streaming=True)
        sample = next(iter(ds))
        print("✅ Successfully accessed one sample.")
        print("   Available keys:", list(sample.keys())[:10])
        meta = {k: sample[k] for k in ["UID", "Institution", "Split", "Fold"] if k in sample}
        print("   Metadata:", meta)
        mods = [k for k in sample.keys() if k.startswith("Image_")]
        print("   Modalities present:", mods)
        print("\n🎉 Access check passed — you can safely start full dataset download.")
    except Exception as e:
        print(f"❌ Error while streaming data: {type(e).__name__}: {e}")
        print("👉 若为 401/403 错误，请回到数据集页面重新同意使用条款。")

if __name__ == "__main__":
    main()
