from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from script import DQN

app = FastAPI()

class State(BaseModel):
    diamonds_collected: int
    played_cards: list


# Load the model (adjust the path as necessary)
model = torch.load('diamant_model.pth')
model.eval()  # Set the model to evaluation mode

@app.post("/predict/")
async def predict(state_model : State):

    dangers_drawed = [card['value'] for card in state_model.played_cards if card['type'] == 'Danger']

    def load_data():
        return [state_model.diamonds_collected] + [int(danger in dangers_drawed) for danger in ['Araignée', 'Pierre', 'Lave', 'Serpent', 'Pique']]

    state = load_data()

    # Validate the state length
    expected_state_length = 6
    if len(state) != expected_state_length:
        raise HTTPException(status_code=400, detail=f"State must have {expected_state_length} elements.")

    try:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get action from model
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        return {"action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
