import streamlit as st
from pathlib import Path
from PIL import Image
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData

import SessionState


def rescale(img):
    width = 800
    height = 400
    w, h = img.size
    k_w = width / w
    k_h = height / h
    mult = min(k_w, k_h)
    new_h = int(h * mult)
    new_w = int(w * mult)
    return img.resize((new_w, new_h))


def main():
    state = SessionState.get(
        ground_truth=0,
        detected_patterns=0,
        true_positive=0,
        image_num=0,
        problems=[],
        dataset_name=None
    )
    st.title('Finder Pattern Detector Tester')
    if state.dataset_name is None:
        output_dir = Path(__file__).parent.parent / "output"
        datasets = [dataset.name for dataset in output_dir.iterdir()]
        dataset_name = st.selectbox('Select Dataset', datasets)
        if st.button("CONTINUE"):
            state.dataset_name = dataset_name
            raise RerunException(RerunData())

    else:
        dataset_path = Path(__file__).parent.parent / "output" / state.dataset_name
        files = [path for path in dataset_path.iterdir()]
        if state.image_num < len(files):
            file = files[state.image_num]
            st.markdown(f"Dataset: {state.dataset_name}")
            st.markdown(f"Image: {file.name}")
            st.image(rescale(Image.open(file)))
            ground_truth = st.number_input("Ground Truth Number of Finder Patterns", 0)
            detected_patterns = st.number_input("Total Number of Detected Finder Patterns", 0)
            true_positive = st.number_input("Number of Correctly Detected Finder Patterns", 0)
            st.markdown(f"Images to score: {len(files) - state.image_num}")

            if st.button("NEXT"):
                state.ground_truth += ground_truth
                state.detected_patterns += detected_patterns
                state.true_positive += true_positive
                state.image_num += 1
                if (detected_patterns != true_positive) or (true_positive != ground_truth):
                    state.problems.append(file.name)
                raise RerunException(RerunData())
        else:
            if state.ground_truth == 0:
                st.markdown("No images found!")
            elif state.detected_patterns == 0:
                st.markdown("No patterns were detected!")
            else:
                precision = state.true_positive / state.detected_patterns
                recall = state.true_positive / state.ground_truth
                st.markdown(f'**Number of images:** {len(files)}')
                st.markdown(f'**Precision:** {precision:.2f}')
                st.markdown(f'**Recall:** {recall:.2f}')
                if len(state.problems) > 0:
                    st.markdown(f"**Images with problems:**")
                    for img in state.problems:
                        st.markdown(f"- **{img}**")

                st.markdown(f'Refresh the page to evaluate another dataset')


if __name__ == "__main__":
    main()
