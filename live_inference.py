import time
import requests

def main():
    print("Starting Live Inference with Local Qwen Model...")
    base_url = "http://localhost:8000"
    
    # 1. Reset the environment to a random incident
    print("Resetting environment to random_incident...")
    res = requests.post(f"{base_url}/reset", json={"task_name": "random_incident"})
    if res.status_code != 200:
        print(f"Failed to reset: {res.text}")
        return
        
    while True:
        # 2. Check if we are done
        state_res = requests.get(f"{base_url}/state").json()
        if state_res["state"]["is_resolved"]:
            print("\n🎉 Incident Resolved!")
            break
        if state_res["state"]["step_count"] >= 30:
            print("\n❌ Max steps reached without resolution.")
            break
            
        print(f"\nStep {state_res['state']['step_count']} - Asking Local Qwen Model...")
        
        # 3. Ask the backend to run the local Qwen model prediction
        pred_res = requests.post(f"{base_url}/predict", json={"adapter_path": "trained_model_0p5b_v2"})
        if pred_res.status_code != 200:
            print(f"Prediction failed! Is your backend running? Error: {pred_res.text}")
            break
            
        action_data = pred_res.json().get("parsed_action")
        valid_actions = ["inspect_logs", "inspect_metrics", "restart_service", "scale_service", "rollback", "clear_cache", "escalate", "do_nothing", "write_runbook"]
        
        if not action_data or action_data.get("action_type") not in valid_actions:
            print(f"Model generated an invalid action. Falling back to do_nothing. Raw output: {pred_res.json().get('raw_response')}")
            action_data = {"action_type": "do_nothing"}
            
        action_type = action_data.get("action_type")
        service = action_data.get("service_name", "")
        print(f"🤖 Qwen Action: {action_type} {service}")
        
        # 4. Execute the action
        step_res = requests.post(f"{base_url}/step", json={"action": action_data})
        if step_res.status_code != 200:
            print(f"Failed to step: {step_res.text}")
            break
            
        # Sleep so you can watch the dashboard update!
        time.sleep(2)

if __name__ == "__main__":
    main()
