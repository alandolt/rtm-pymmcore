# rtm-pymmcore

**Real-time feedback control microscopy using `pymmcore` as an interface.**

## Overview

This repository provides a framework for real-time cell segmentation and feature extraction during live microscopy experiments, leveraging the `pymmcore` library for microscope control. This approach enables immediate feedback control, allowing for dynamic adjustments based on observed cellular behavior. For instance, in systems with spatial stimulation capabilities, stimulation areas can be automatically defined based on real-time cell morphology analysis.

Key applications include:

* **Real-time feedback control:** Dynamically adjusting experimental parameters based on live cell analysis.
* **Streamlined experiments:** Minimizing post-processing requirements by performing analysis during acquisition.
* **Dynamic stimulation:** Targeting specific cells or regions based on their morphology or behavior.

## Key Features

* **Integration with `pymmcore-plus`:** Utilizes the advanced features of `pymmcore-plus` for robust microscope control and image acquisition.
* **Modular image processing pipeline:** Allows for flexible integration of different segmentation, stimulation, feature extraction, and tracking modules.
* **External segmentation engine support:** Option to offload computationally intensive segmentation tasks to an external server using `imaging-server-kit`.
* **Comprehensive data handling:** Stores acquisition positions, stimulation parameters, and single-cell results in structured dataframes.
* **User-friendly visualization and analysis tools:** Provides scripts for visualizing individual fields of view and generating reports.
* **Jupyter Notebook-based workflow:** Enables step-by-step execution and easy customization of the pipeline.

## Workflow

The real-time microscopy pipeline follows these main steps:

1.  **Defining Acquisition Positions & Stimulation Parameters:**
    * A Pandas DataFrame is created to store the coordinates of acquisition positions. These positions can be interactively selected using GUI widgets provided by `pymmcore-plus`.
    * Stimulation parameters, such as pulse duration and intensity, are defined and configured within the Jupyter notebook environment.

2.  **Image Acquisition & Processing:**
    * The acquisition position DataFrame is passed to the core of the `rtm-pymmcore` pipeline.
    * The pipeline orchestrates image acquisition, followed by a series of processing steps:
        * **Segmentation:** Identifying individual cells within the acquired images.
        * **Stimulation:** Applying a defined stimulation pattern based on segmentation results or other criteria.
        * **Feature Extraction :** Measuring relevant characteristics of the segmented cells (e.g., size, shape, intensity).
        * **Tracking:** Following individual cells over time across multiple frames.
    * Optionally, an external segmentation engine running via `imaging-server-kit` can be utilized for segmentation.
    * Single-cell results, including extracted features and tracking information, are stored in a structured Pandas DataFrame.

3.  **Data Visualization & Analysis:**
    * The `viewer` script enables visualization of individual fields of view (FOVs) overlaid with stimulation conditions and segmentation masks.
    * The `data_analysis_plotting` scripts provide tools for generating reports and performing further analysis on the collected data.

### Image Processing Steps

The image processing pipeline is designed with modularity in mind. Each step is implemented as an independent module, allowing you to load and utilize specific modules without requiring dependencies for others. This is particularly useful when you only need a subset of the processing steps (e.g., using only `stardist` for segmentation without installing `cellpose`).

* **Image Segmentation:**
    * This step focuses on identifying and delineating individual cells within the acquired images.
    * The desired segmentation engine(s) and their specific parameters are configured in the Jupyter notebook using a list of dictionaries.
    ```python
    # Example of a segmentation engine configuration
    segmentators = [
        {
            "name": "labels",
            "class": SegmentorStardist(),
            "use_channel": 0,
            "save_tracked": True,
        },
      {
            "name": "cell_body",
            "class": SegmentorCellpose(model="cyto3", diameter=75, flow_threshold=0.4),
            "use_channel": 1,
            "save_tracked": True,
        },
    ]
    ```

* **Stimulation Engine:**
    * This optional step allows for applying targeted stimulation based on the segmentation results.
    * It can utilize a Digital Micromirror Device (DMD) for structured illumination, enabling stimulation of specific cells or regions defined by the segmentation masks.
    * Alternatively, if no DMD is available, full field of view stimulation will be performed.
    * The stimulation mask generated from the segmentation is sent to the DMD or microscope control system to apply the desired stimulation pattern.

    ```python
    stimulator = StimPercentageOfCell()
    ```

* **Feature Extraction:**
    * In this step, quantitative features are extracted from the segmented cells.
    * The pipeline currently everages the `scikit-image` library for various feature extraction functionalities.
    * Two example feature extractors are provided:
        * `SimpleFE`: Extracts basic features like cell position and area. You can specify which segmentation mask to use for feature extraction.
        * A more advanced feature extractor for analyzing ERK activity based on an ERK-KTR translocation biosensor. This extractor utilizes both the segmented cell images and the original acquired image for each time point.

    ```python
    feature_extractor = SimpleFE("labels") # Extract features from the segmentation named "celllabelsose_cyto3"
    # Example of a more complex feature extractor
    # feature_extractor = FE_ErkKtr()
    ```

* **Tracker:**
    * This module enables tracking individual cells over time across consecutive image frames.
    * It utilizes the `trackpy` library as its backend for robust cell tracking.
    * The tracker analyzes the segmented cells to establish correspondences between frames, allowing for the study of cell movement and dynamics.

    ```python
    tracker = TrackerTrackpy(search_range=30) # Example: Set the maximum search distance for tracking to 30 pixels
    ```

**Important Considerations:**

* You can freely combine any of the provided modules to create a customized image processing pipeline tailored to your specific experimental needs.
* It is also possible to develop and integrate your own custom modules into the pipeline.
* If a particular processing step is not required for your experiment, you can simply set the corresponding module to `None` in your Jupyter notebook configuration.

### Microscope Setup

The `rtm-pymmcore` workflow relies on a **uManager configuration file** to define the microscope hardware setup. This configuration file is essential for controlling the microscope and acquiring images. The configuration file must include:

* A `setup` group containing a preset named `Startup`. This preset will be automatically executed when the script initializes, typically configuring essential hardware components.
* Additional presets for each fluorophore used in the experiment. These presets should define the appropriate settings for filters, lasers, and other optical elements required for imaging each fluorescent channel (e.g., presets named after the fluorophores like `GFP`, `mCherry`, etc.).

### Running the Script

The workflow is designed to be executed step by step using Jupyter notebooks. This allows for interactive control and monitoring of the experiment. Currently, three example workflows are provided:

* **`00_NoStim_MMDemo.ipynb`**: This notebook demonstrates how to run the pipeline without any stimulation. It focuses on cell tracking and feature extraction. This is an excellent starting point for users to familiarize themselves with the basic structure of the pipeline and the functionality of different modules. It utilizes a simulated Micro-Manager demo microscope and loads example images from the `test_exp_data` folder.

* **`01_ERK-KTR_full_fov_stimulation.ipynb`** and **`01_ERK-KTR_full_fov_stimulation_Jungfrau.ipynb`**: These notebooks showcase a full field of view stimulation experiment targeting cells expressing an optogenetic actuator, such as FGFR1. The translocation of the ERK-KTR biosensor is measured in all individual cells as a readout of the stimulation response. Two versions of this script are available:
    * The first version (`01_ERK-KTR_full_fov_stimulation.ipynb`) is designed to work with the Micro-Manager demo microscope, simulating a real experiment using images from the `test_exp_data` folder. This allows users to test and understand the workflow without needing access to a physical microscope.
    * The second version (`01_ERK-KTR_full_fov_stimulation_Jungfrau.ipynb`) is specifically configured for use with our Ti Eclipse microscope (named "Jungfrau"), demonstrating the pipeline's application on a real experimental setup.

* **`02_CellMigration.ipynb`** and **`02_CellMigration_Niesen.ipynb`**: These notebooks illustrate how the pipeline can be used for studying directed cell migration. In this example, only the front part of the migrating cells is selectively stimulated with light using structured illumination. Similar to the previous example, two versions are provided:
    * `02_CellMigration.ipynb` is intended for use with the Micro-Manager demo microscope.
    * `02_CellMigration_Niesen.ipynb` is configured for our Ti2 microscope equipped with a DMD for structured illumination (named "Niesen").

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/rtm-pymmcore.git](https://github.com/pertzlab/rtm-pymmcore.git)
    cd rtm-pymmcore
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Install optional dependencies if needed (e.g., for specific segmentation engines)
    # pip install cellpose stardist trackpy scikit-image
    ```

## Usage

The recommended way to use this pipeline is by running the provided Jupyter notebooks. Each notebook demonstrates a specific workflow.

1.  Ensure you have a Micro-Manager configuration file set up according to the requirements outlined in the "Microscope Setup" section.
2.  Open the desired Jupyter notebook (e.g., `00_NoStim_MMDemo.ipynb`).
3.  Follow the instructions within the notebook to configure the experiment parameters and execute the pipeline step by step.

## Contributing

Contributions to this repository are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
