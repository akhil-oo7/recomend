import random
import pandas as pd
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Reward options
reward_options = ["Coupon10", "Cashback", "GiftCard", "TShirt", "Voucher"]

# Sample attributes
genders = ["Male", "Female", "Other"]
locations = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
income_levels = ["Low", "Medium", "High"]

# Generate data
data = []

for i in range(10000):
    age = random.randint(18, 60)
    gender = random.choice(genders)
    location = random.choice(locations)
    income = random.choice(income_levels)
    activity_score = random.randint(0, 100)

    # Weighted reward assignment (based on activity score or other factors)
    reward_count = random.randint(1, 3)
    reward_choices = random.sample(reward_options, reward_count)

    data.append({
        "user_id": i + 1,
        "age": age,
        "gender": gender,
        "location": location,
        "income_level": income,
        "activity_score": activity_score,
        "rewards": ";".join(reward_choices)  # ✅ This is the column required for training
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("data/UserRewardSim.csv", index=False)

print("✅ Dataset generated at data/UserRewardSim.csv")
