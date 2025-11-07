import numpy as np

def bat_algorithm(model, val_generator, population_size=10, max_iter=10):
    original_weights = model.get_weights()
    population = [original_weights.copy() for _ in range(population_size)]
    fitness = [float('inf')] * population_size
    best_weights = None
    best_fitness = float('inf')

    for iteration in range(max_iter):
        for i in range(population_size):
            candidate_weights = [
                w + np.random.uniform(-0.1, 0.1, size=w.shape)
                for w in population[i]
            ]
            model.set_weights(candidate_weights)
            val_loss, _ = model.evaluate(val_generator, verbose=0)

            if val_loss < fitness[i]:
                fitness[i] = val_loss
                population[i] = candidate_weights

            if val_loss < best_fitness:
                best_fitness = val_loss
                best_weights = candidate_weights

        print(f"Iteration {iteration+1}/{max_iter} | Best Validation Loss: {best_fitness}")

    model.set_weights(best_weights)
    return model
