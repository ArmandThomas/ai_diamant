from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from script import load_model

app = FastAPI()

class State(BaseModel):
    diamonds_collected: int
    played_cards: list


# Load the model (adjust the path as necessary)
model = load_model()
model.eval()  # Set the model to evaluation mode

@app.post("/predict/")
async def predict(state_model: State):

    def load_data():
        array = [state_model.diamonds_collected]
        for card in state_model.played_cards:
            if card['type'] == 'Trésor':
                array.append(1)
            elif card['type'] == 'Relique':
                array.append(7)
            elif card['type'] == 'Danger':
                if card['value'] == 'Araignée':
                    array.append(2)
                elif card['value'] == 'Lave':
                    array.append(3)
                elif card['value'] == 'Pierre':
                    array.append(4)
                elif card['value'] == 'Serpent':
                    array.append(5)
                elif card['value'] == 'Pique':
                    array.append(6)

        for cards_not_played in range(35 - len(state_model.played_cards)):
            array.append(0)

        return array

    state = load_data()

    # Validate the state length
    expected_state_length = 36
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
