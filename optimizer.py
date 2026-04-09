import numpy as np
import random
import pandas as pd

def optimize_mix(model, scaler, base_inputs, generations=20, pop_size=12):

    waste_cols = [
        "Pig Manure (kg)", "Kitchen Food Waste (kg)", "Chicken Litter (kg)",
        "Cassava (kg)", "Bagasse Feed (kg)", "Energy Grass (kg)",
        "Banana Shafts (kg)", "Alcohol Waste (kg)",
        "Municipal Residue (kg)", "Fish Waste (kg)"
    ]

    total_waste = sum([base_inputs[w] for w in waste_cols])

    def create_population():
        pop = []
        for _ in range(pop_size):
            vals = np.random.rand(len(waste_cols))
            vals /= vals.sum()
            pop.append(vals * total_waste)
        return np.array(pop)

    def evaluate_population(population):
        rows = []

        for ind in population:
            temp = base_inputs.copy()
            for i, w in enumerate(waste_cols):
                temp[w] = ind[i]
            rows.append(temp)

        df = pd.DataFrame(rows)
        scaled = scaler.transform(df)

        preds = model.predict(scaled, verbose=0).flatten()
        return preds

    # Initial population
    population = create_population()

    for _ in range(generations):

        scores = evaluate_population(population)

        # Select top half
        idx = np.argsort(scores)[::-1][:pop_size // 2]
        selected = population[idx]

        # Create next generation
        new_population = []

        while len(new_population) < pop_size:
            p1, p2 = selected[random.randint(0, len(selected)-1)], selected[random.randint(0, len(selected)-1)]

            alpha = np.random.rand(len(waste_cols))
            child = alpha * p1 + (1 - alpha) * p2

            # mutation
            if random.random() < 0.2:
                i = np.random.randint(len(waste_cols))
                child[i] *= np.random.uniform(0.8, 1.2)

            # normalize
            child = np.maximum(child, 0)
            child /= child.sum()
            child *= total_waste

            new_population.append(child)

        population = np.array(new_population)

    # Final evaluation
    final_scores = evaluate_population(population)
    best_idx = np.argmax(final_scores)

    best = population[best_idx]

    best_mix = {w: round(best[i], 2) for i, w in enumerate(waste_cols)}
    best_biogas = round(final_scores[best_idx], 2)

    return best_mix, best_biogas