# Example of using deepforest to predict airborne birds

from deepforest import main
from deepforest import visualize
import matplotlib.pyplot as plt

m = main.deepforest()
m.load_model("Weecology/deepforest-bird")

image_path="/Users/benweinstein/Downloads/example_airborne_birds/5.jpg"
predictions = m.predict_tile(image_path, patch_size=600, patch_overlap=0)
# visualize the predictions

visualize.plot_results(predictions)
plt.show()

high_confidence = predictions[predictions.score>0.2]
high_confidence.root_dir = "/Users/benweinstein/Downloads/example_airborne_birds"

visualize.plot_results(high_confidence)