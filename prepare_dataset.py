import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("https://github.com/AdonaiVera/bddoia-fiftyone", split="validation", dataset_name="workshop_iccv_2025_original", persistent=True)
filtered_dataset = dataset.match({"weather.label": {"$ne": "undefined"}, "scene.label": {"$ne": "undefined"}, "timeofday.label": {"$ne": "undefined"}})
weather_labels = filtered_dataset.distinct("weather.label")
scene_labels = filtered_dataset.distinct("scene.label")
timeofday_labels = filtered_dataset.distinct("timeofday.label")
filtered_dataset.shuffle()

selected_ids = set()
subset_samples = []
valid_combinations = []
for weather in weather_labels:
    for scene in scene_labels:
        for timeofday in timeofday_labels:
            view = filtered_dataset.match({"weather.label": weather, "scene.label": scene, "timeofday.label": timeofday})
            if len(view) > 0:
                valid_combinations.append((weather, scene, timeofday))

samples_per_combo = 100 // len(valid_combinations)
remaining = 100 % len(valid_combinations)

for i, (weather, scene, timeofday) in enumerate(valid_combinations):
    view = filtered_dataset.match({"weather.label": weather, "scene.label": scene, "timeofday.label": timeofday})
    samples_to_take = samples_per_combo + (1 if i < remaining else 0)
    samples_to_take = min(samples_to_take, len(view))
    for sample in view[:samples_to_take]:
        if sample.id not in selected_ids:
            selected_ids.add(sample.id)
            subset_samples.append(sample)

subset = fo.Dataset("workshop_iccv_2025")
subset.add_samples(subset_samples)
subset.persistent = True