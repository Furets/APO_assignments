import random
import pandas as pd
import plotly.express as px

# Constants
NUM_CARDS = 100
RESOURCE_TYPES = ['machinery', 'basic_ingredients', 'special_ingredients', 'staff_skills', 'stores', 'warehouse']
MAX_LEVEL = 5
REWARD_BASE = 10
COST_BASE = 5

# Strategies Definition with balanced points
strategies = {
    'buy_to_order': {'machinery': 2, 'basic_ingredients': 2, 'special_ingredients': 5, 'staff_skills': 5, 'stores': 1,
                     'warehouse': 1},
    'make_to_order': {'machinery': 3, 'basic_ingredients': 3, 'special_ingredients': 4, 'staff_skills': 3, 'stores': 2,
                      'warehouse': 2},
    'assembly_to_order': {'machinery': 2, 'basic_ingredients': 4, 'special_ingredients': 3, 'staff_skills': 2,
                          'stores': 3, 'warehouse': 3},
    'make_to_stock': {'machinery': 4, 'basic_ingredients': 3, 'special_ingredients': 1, 'staff_skills': 1, 'stores': 4,
                      'warehouse': 5}
}

# Define the costs to buy each type of resource
resource_buying_costs = {
    'machinery': 20,
    'basic_ingredients': 10,
    'special_ingredients': 15,
    'staff_skills': 25,
    'stores': 8,
    'warehouse': 12
}


# Function to assign resources based on strategy
def assign_resources(strategy):
    return strategies[strategy].copy()


# Function to generate a random demand card
def generate_demand_card():
    card = {resource: random.randint(1, MAX_LEVEL) for resource in RESOURCE_TYPES}
    while sum(card.values()) > 20:  # Example threshold for balancing
        card = {resource: random.randint(1, MAX_LEVEL) for resource in RESOURCE_TYPES}
    return card


# Function to calculate reward for satisfying a demand
def calculate_reward(card, resources):
    multiplier = min(
        (resources[resource] / card[resource]) if card[resource] > 0 else float('inf') for resource in RESOURCE_TYPES)
    difficulty_reward = sum(card.values()) * 2  # Example difficulty reward calculation
    return (REWARD_BASE * multiplier) + difficulty_reward


# Function to determine the cost of resource expansion
def expansion_cost(current_level):
    return COST_BASE * current_level


# Function to buy resources based on available funds and strategy
def buy_resources(player_resources, funds, strategy):
    for resource in RESOURCE_TYPES:
        # Determine the quantity to buy based on the strategy's focus on that resource
        focus_level = strategies[strategy][resource]
        quantity_to_buy = (MAX_LEVEL - focus_level)  # Simplified logic for example
        cost_to_buy = resource_buying_costs[resource] * quantity_to_buy

        # Check if there are enough funds to buy the resource
        if funds >= cost_to_buy:
            player_resources[resource] += quantity_to_buy
            funds -= cost_to_buy
    return funds


# Function to simulate a single round of the game
def simulate_round(strategy, demand_cards, player_resources):
    round_reward = 0
    for card in demand_cards:
        if all(player_resources[res] >= card[res] for res in RESOURCE_TYPES):
            round_reward += calculate_reward(card, player_resources)
            # Double reward if resources are double or more than required
            if all(player_resources[res] >= 2 * card[res] for res in RESOURCE_TYPES):
                round_reward *= 2
    return round_reward


# Function to simulate the game
def simulate_game(strategy):
    player_resources = assign_resources(strategy)
    total_reward = 0
    funds = 0  # Initialize funds to zero
    all_cards = [generate_demand_card() for _ in range(NUM_CARDS)]  # Generate all 100 cards
    random.shuffle(all_cards)  # Shuffle the cards

    while all_cards:
        # Draw up to 5 cards per round
        round_cards = all_cards[:5]
        all_cards = all_cards[5:]
        round_reward = simulate_round(strategy, round_cards, player_resources)
        total_reward += round_reward
        funds += round_reward  # Add reward to funds
        funds = buy_resources(player_resources, funds, strategy)  # Buy resources with available funds

    return total_reward, player_resources


# Simulate multiple games and calculate the average ending resources
def simulate_games(num_games):
    results = []
    resource_totals = {strategy: {resource: 0 for resource in RESOURCE_TYPES} for strategy in strategies}
    for strategy in strategies:
        for _ in range(num_games):
            total_reward, final_resources = simulate_game(strategy)
            results.append({'strategy': strategy, 'total_reward': total_reward})
            for resource, amount in final_resources.items():
                resource_totals[strategy][resource] += amount

        # Average the resource totals over all games
        for resource in RESOURCE_TYPES:
            resource_totals[strategy][resource] /= num_games

    return pd.DataFrame(results), resource_totals


# Simulate games and store results in a DataFrame
df_results, resource_averages = simulate_games(100)

# Convert the resource averages to a long-form DataFrame for plotting
resource_averages_df = pd.DataFrame(resource_averages).reset_index().melt(id_vars='index', var_name='Resource',
                                                                          value_name='Average Quantity')
resource_averages_df.rename(columns={'index': 'Strategy'}, inplace=True)

# Create a plot using Plotly
fig = px.bar(resource_averages_df, x='Strategy', y='Average Quantity', color='Resource',
             title='Average Ending Resource Quantities by Strategy')
fig.show()
