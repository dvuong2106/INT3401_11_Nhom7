import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import backend
from models import (
    Attention,
    Convolve,
    DigitClassificationModel,
    DigitConvolutionalModel,
    LanguageIDModel,
    PerceptronModel,
    RegressionModel,
)
from train import (
    Train_DigitConvolution,
    train_digitclassifier,
    train_languageid,
    train_perceptron,
    train_regression,
)


@st.cache_resource(show_spinner=False)
def train_q1_model():
    backend.use_graphics = False
    model = PerceptronModel(3)
    dataset = backend.PerceptronDataset(model)
    initial_weights = model.get_weights().detach().cpu().numpy().reshape(-1).copy()
    train_perceptron(model, dataset)

    weights = model.get_weights().detach().cpu().numpy().reshape(-1)
    predictions = np.where(dataset.x @ weights >= 0, 1, -1)
    accuracy = float(np.mean(predictions == dataset.y.reshape(-1)))
    return model, dataset, initial_weights, accuracy


@st.cache_resource(show_spinner=False)
def train_q2_model():
    backend.use_graphics = False
    model = RegressionModel()
    dataset = backend.RegressionDataset(model)
    train_regression(model, dataset)
    model.eval()

    with torch.no_grad():
        features = torch.tensor(dataset.x, dtype=torch.float32)
        predictions = model(features).reshape(-1).cpu().numpy()
    mse = float(np.mean((predictions - dataset.y.reshape(-1)) ** 2))
    return model, dataset, predictions, mse


@st.cache_resource(show_spinner=False)
def train_q3_model():
    backend.use_graphics = False
    model = DigitClassificationModel()
    dataset = backend.DigitClassificationDataset(model)
    train_digitclassifier(model, dataset)
    model.eval()
    return model, dataset


@st.cache_resource(show_spinner=False)
def train_q4_model():
    backend.use_graphics = False
    model = LanguageIDModel()
    dataset = backend.LanguageIDDataset(model)
    train_languageid(model, dataset)
    model.eval()
    return model, dataset


@st.cache_resource(show_spinner=False)
def train_q5_model():
    backend.use_graphics = False
    model = DigitConvolutionalModel()
    dataset = backend.DigitClassificationDataset2(model)
    Train_DigitConvolution(model, dataset)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(dataset.test_images, dtype=torch.float32))
        accuracy = float(
            (logits.argmax(dim=1).cpu().numpy() == dataset.test_labels).mean()
        )
    return model, dataset, accuracy


@st.cache_resource(show_spinner=False)
def create_q6_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Attention(layer_size=4, block_size=8)
    model.eval()
    return model


def parse_embedding_matrix(text: str, expected_rows: int) -> torch.Tensor:
    rows = [
        [float(value.strip()) for value in line.split(",")]
        for line in text.strip().splitlines()
        if line.strip()
    ]
    if len(rows) != expected_rows:
        raise ValueError(f"Cần nhập đúng {expected_rows} dòng embedding.")
    if any(len(row) != 4 for row in rows):
        raise ValueError("Mỗi dòng cần có đúng 4 giá trị, phân cách bằng dấu phẩy.")
    if not np.isfinite(rows).all():
        raise ValueError("Embedding chỉ được chứa các số hữu hạn.")
    return torch.tensor(rows, dtype=torch.float32)


def preprocess_image(image: Image.Image) -> np.ndarray | None:
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale)

    if pixels.mean() > 127:
        grayscale = ImageOps.invert(grayscale)

    digit = np.asarray(grayscale, dtype=np.float32) / 255.0
    bounds = np.argwhere(digit > 0.1)
    if bounds.size == 0:
        return None

    top, left = bounds.min(axis=0)
    bottom, right = bounds.max(axis=0) + 1
    cropped = Image.fromarray(
        np.uint8(digit[top:bottom, left:right] * 255)
    )
    resized = ImageOps.contain(
        cropped,
        (20, 20),
        method=Image.Resampling.LANCZOS,
    )

    centered = Image.new("L", (28, 28), color=0)
    centered.paste(
        resized,
        ((28 - resized.width) // 2, (28 - resized.height) // 2),
    )
    return np.asarray(centered, dtype=np.float32).reshape(784) / 255.0


def preprocess_canvas(image_data: np.ndarray) -> np.ndarray | None:
    image = Image.fromarray(np.uint8(image_data), mode="RGBA")
    background = Image.new("RGBA", image.size, "white")
    composite = Image.alpha_composite(background, image).convert("RGB")
    return preprocess_image(composite)


def draw_decision_boundary(ax, weights: np.ndarray, label: str, **kwargs) -> None:
    if abs(weights[1]) > 1e-8:
        x_values = np.linspace(-3.5, 3.5, 200)
        y_values = -(weights[0] * x_values + weights[2]) / weights[1]
        ax.plot(x_values, y_values, label=label, **kwargs)
    elif abs(weights[0]) > 1e-8:
        ax.axvline(-weights[2] / weights[0], label=label, **kwargs)


st.set_page_config(page_title="Mô phỏng Q1–Q6", page_icon="🧠", layout="wide")
st.title("Mô phỏng Machine Learning — Q1 đến Q6")
st.write(
    "Trực quan hóa các mô hình Perceptron, neural network, CNN, RNN và attention."
)

q1_tab, q2_tab, q3_tab, q4_tab, q5_tab, q6_tab, info_tab = st.tabs(
    [
        "Mô phỏng Q1",
        "Mô phỏng Q2",
        "Mô phỏng Q3",
        "Mô phỏng Q4",
        "Mô phỏng Q5",
        "Mô phỏng Q6",
        "Thông tin",
    ]
)

with q1_tab:
    st.subheader("Q1 — Perceptron phân loại nhị phân")
    st.write(
        "Mô hình học đường ranh giới để phân loại 500 điểm thành hai lớp "
        "trong mặt phẳng."
    )
    if not st.session_state.get("q1_model_ready", False):
        if st.button("Huấn luyện Perceptron", type="primary"):
            with st.spinner("Đang huấn luyện Perceptron trên dữ liệu Q1..."):
                model, dataset, initial_weights, accuracy = train_q1_model()
            st.session_state.q1_model_ready = True
            st.success(f"Huấn luyện xong — độ chính xác: {accuracy:.2%}")
        else:
            st.info("Huấn luyện mô hình để xem ranh giới phân loại.")

    if st.session_state.get("q1_model_ready", False):
        model, dataset, initial_weights, accuracy = train_q1_model()
        weights = model.get_weights().detach().cpu().numpy().reshape(-1)
        points = dataset.x
        labels = dataset.y.reshape(-1)

        figure, axis = plt.subplots()
        axis.scatter(
            points[labels == -1, 0],
            points[labels == -1, 1],
            label="Lớp −1",
            alpha=0.65,
            s=20,
        )
        axis.scatter(
            points[labels == 1, 0],
            points[labels == 1, 1],
            label="Lớp +1",
            alpha=0.65,
            s=20,
        )
        draw_decision_boundary(
            axis,
            np.array([1.0, 2.0, -1.0]),
            "Ranh giới đúng",
            color="green",
            linestyle=":",
        )
        draw_decision_boundary(
            axis,
            initial_weights,
            "Ban đầu",
            color="gray",
            linestyle="--",
        )
        draw_decision_boundary(
            axis,
            weights,
            "Perceptron đã học",
            color="black",
        )
        axis.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), xlabel="x₁", ylabel="x₂")
        axis.grid(alpha=0.2)
        axis.legend()
        st.pyplot(figure)
        plt.close(figure)

        first_metric, second_metric = st.columns(2)
        first_metric.metric("Độ chính xác trên dữ liệu Q1", f"{accuracy:.2%}")
        second_metric.write(
            "Trọng số đã học: "
            f"`[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]`"
        )

with q2_tab:
    st.subheader("Q2 — Hồi quy phi tuyến")
    st.write(
        "Mạng neural học xấp xỉ hàm sin từ các điểm mẫu trong khoảng "
        "[−2π, 2π]."
    )
    if not st.session_state.get("q2_model_ready", False):
        if st.button("Huấn luyện mô hình hồi quy", type="primary"):
            with st.spinner("Đang huấn luyện mô hình Q2..."):
                model, dataset, predictions, mse = train_q2_model()
            st.session_state.q2_model_ready = True
            st.success(f"Huấn luyện xong — MSE trên tập mẫu: {mse:.5f}")
        else:
            st.info("Huấn luyện mô hình để xem đường cong sin và dự đoán.")

    if st.session_state.get("q2_model_ready", False):
        model, dataset, predictions, mse = train_q2_model()
        sorted_indices = np.argsort(dataset.x.reshape(-1))
        x_values = dataset.x.reshape(-1)[sorted_indices]
        true_values = dataset.y.reshape(-1)[sorted_indices]
        predicted_values = predictions[sorted_indices]

        figure, axis = plt.subplots()
        axis.plot(x_values, true_values, label="sin(x)", color="blue")
        axis.plot(
            x_values,
            predicted_values,
            label="Mô hình dự đoán",
            color="orange",
        )
        axis.set(xlabel="x", ylabel="y", xlim=(-2 * np.pi, 2 * np.pi))
        axis.grid(alpha=0.2)
        axis.legend()
        st.pyplot(figure)
        plt.close(figure)

        st.metric("MSE trên 200 điểm mẫu", f"{mse:.5f}")
        selected_x = st.slider(
            "Chọn x để so sánh giá trị",
            min_value=float(-2 * np.pi),
            max_value=float(2 * np.pi),
            value=0.0,
            step=0.05,
        )
        with torch.no_grad():
            predicted_value = float(
                model(torch.tensor([[selected_x]], dtype=torch.float32)).item()
            )
        actual_value = float(np.sin(selected_x))
        actual_column, predicted_column = st.columns(2)
        actual_column.metric("sin(x)", f"{actual_value:.4f}")
        predicted_column.metric("Mô hình dự đoán", f"{predicted_value:.4f}")

with q3_tab:
    st.subheader("Q3 — Phân loại chữ số viết tay")
    st.write(
        "Huấn luyện mô hình Q3 trên MNIST, sau đó thử phân loại chữ số do bạn vẽ "
        "hoặc ảnh mẫu."
    )
    model = None
    dataset = None
    if st.session_state.get("q3_model_ready", False):
        model, dataset = train_q3_model()
    else:
        st.info("Cần huấn luyện mô hình trước khi mô phỏng.")
        if st.button("Huấn luyện mô hình Q3", type="primary"):
            with st.spinner("Đang huấn luyện trên MNIST. Lần đầu có thể mất vài phút..."):
                model, dataset = train_q3_model()
            st.session_state.q3_model_ready = True
            st.success(
                f"Huấn luyện xong — độ chính xác validation: "
                f"{dataset.get_validation_accuracy():.2%}"
            )

    if model is None and st.session_state.get("q3_model_ready", False):
        model, dataset = train_q3_model()

    if model is not None and dataset is not None:
        input_mode = st.radio(
            "Chọn đầu vào",
            ["Vẽ chữ số", "Ảnh mẫu MNIST", "Tải ảnh lên"],
            horizontal=True,
        )

        input_image = None
        expected_label = None

        if input_mode == "Vẽ chữ số":
            st.caption("Vẽ một chữ số từ 0 đến 9 trên vùng vẽ.")
            canvas = st_canvas(
                fill_color="#ffffff",
                stroke_width=18,
                stroke_color="#000000",
                background_color="#ffffff",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="q3_digit_canvas",
            )
            if canvas.image_data is not None:
                input_image = preprocess_canvas(canvas.image_data)
        elif input_mode == "Ảnh mẫu MNIST":
            sample_index = st.slider(
                "Chọn ảnh kiểm tra",
                min_value=0,
                max_value=len(dataset.test_images) - 1,
                value=0,
            )
            input_image = dataset.test_images[sample_index]
            expected_label = int(dataset.test_labels[sample_index])
            st.image(
                input_image.reshape(28, 28),
                caption=f"Nhãn đúng: {expected_label}",
                width=180,
                clamp=True,
            )
        else:
            uploaded_image = st.file_uploader(
                "Tải ảnh chữ số (PNG hoặc JPEG)",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_image is not None:
                image = Image.open(uploaded_image).convert("RGB")
                input_image = preprocess_image(image)

        if st.button("Phân loại chữ số", type="primary"):
            if input_image is None:
                st.warning("Hãy vẽ, chọn hoặc tải lên ảnh có chứa một chữ số.")
            else:
                features = torch.tensor(
                    input_image,
                    dtype=torch.float32,
                ).reshape(1, 784)
                with torch.no_grad():
                    probabilities = torch.softmax(model(features), dim=1)[0]

                prediction = int(probabilities.argmax().item())
                confidence = float(probabilities[prediction].item())
                st.metric(
                    "Mô hình dự đoán",
                    prediction,
                    f"Độ tin cậy {confidence:.1%}",
                )
                if expected_label is not None:
                    if prediction == expected_label:
                        st.success("Mô hình dự đoán đúng.")
                    else:
                        st.warning(f"Nhãn đúng là {expected_label}.")
                st.bar_chart(
                    {"Xác suất": probabilities.tolist()},
                    x_label="Chữ số",
                    y_label="Xác suất",
                )

with q4_tab:
    st.subheader("Q4 — Nhận diện ngôn ngữ bằng RNN")
    st.write(
        "Mô hình đọc lần lượt từng ký tự của một từ và dự đoán ngôn ngữ "
        "trong năm ngôn ngữ của bộ dữ liệu."
    )
    if not st.session_state.get("q4_model_ready", False):
        st.info("Huấn luyện RNN trên bộ dữ liệu nhận diện ngôn ngữ để bắt đầu.")
        if st.button("Huấn luyện mô hình Q4", type="primary"):
            with st.spinner("Đang huấn luyện RNN trên dữ liệu Q4..."):
                model, dataset = train_q4_model()
            st.session_state.q4_model_ready = True
            st.success(
                f"Huấn luyện xong — độ chính xác validation: "
                f"{dataset.get_validation_accuracy():.2%}"
            )

    if st.session_state.get("q4_model_ready", False):
        model, dataset = train_q4_model()
        word = st.text_input("Nhập một từ", value="hello", max_chars=20).lower()
        if st.button("Nhận diện ngôn ngữ", type="primary"):
            chars = dataset.chars.tolist()
            unknown_chars = sorted(set(word) - set(chars))
            if not word:
                st.warning("Hãy nhập một từ để phân loại.")
            elif unknown_chars:
                st.warning(
                    "Ký tự không có trong bảng chữ cái của mô hình: "
                    + ", ".join(unknown_chars)
                )
            else:
                char_indices = [chars.index(char) for char in word]
                features = torch.eye(len(chars), dtype=torch.float32)[char_indices]
                with torch.no_grad():
                    probabilities = torch.softmax(model(features), dim=-1)
                prediction = int(probabilities.argmax().item())
                st.metric(
                    "Ngôn ngữ dự đoán",
                    str(dataset.language_names[prediction]),
                    f"Độ tin cậy {probabilities[prediction].item():.1%}",
                )
                st.bar_chart(
                    {
                        "Ngôn ngữ": [str(name) for name in dataset.language_names],
                        "Xác suất": probabilities.cpu().tolist(),
                    },
                    x="Ngôn ngữ",
                    y="Xác suất",
                )

with q5_tab:
    st.subheader("Q5 — CNN và phép tích chập")
    st.write(
        "CNN áp dụng kernel 3×3 lên ảnh MNIST, sau đó dùng các lớp fully "
        "connected để phân loại chữ số."
    )
    if not st.session_state.get("q5_model_ready", False):
        st.info("Huấn luyện CNN trên dữ liệu Q5 để xem dự đoán và feature map.")
        if st.button("Huấn luyện mô hình Q5", type="primary"):
            with st.spinner("Đang huấn luyện CNN trên dữ liệu Q5..."):
                model, dataset, test_accuracy = train_q5_model()
            st.session_state.q5_model_ready = True
            st.success(f"Huấn luyện xong — độ chính xác tập kiểm tra: {test_accuracy:.2%}")

    if st.session_state.get("q5_model_ready", False):
        model, dataset, test_accuracy = train_q5_model()
        sample_index = st.slider(
            "Chọn ảnh kiểm tra Q5",
            min_value=0,
            max_value=len(dataset.test_images) - 1,
            value=0,
            key="q5_sample_index",
        )
        image = dataset.test_images[sample_index]
        label = int(dataset.test_labels[sample_index])
        features = torch.tensor(image, dtype=torch.float32).reshape(1, 784)
        with torch.no_grad():
            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)[0]
            activation = Convolve(
                torch.tensor(image, dtype=torch.float32).reshape(28, 28),
                model.convolution_weights,
            )
        prediction = int(probabilities.argmax().item())

        image_column, activation_column = st.columns(2)
        image_column.image(
            image.reshape(28, 28),
            caption=f"Ảnh MNIST — nhãn đúng: {label}",
            width=200,
            clamp=True,
        )
        activation_figure, activation_axis = plt.subplots()
        activation_axis.imshow(activation.cpu().numpy(), cmap="viridis")
        activation_axis.set_title("Feature map sau tích chập 3×3")
        activation_axis.axis("off")
        activation_column.pyplot(activation_figure)
        plt.close(activation_figure)

        prediction_column, accuracy_column = st.columns(2)
        prediction_column.metric(
            "Mô hình dự đoán",
            prediction,
            f"Độ tin cậy {probabilities[prediction].item():.1%}",
        )
        accuracy_column.metric("Độ chính xác tập kiểm tra Q5", f"{test_accuracy:.2%}")
        if prediction == label:
            st.success("Mô hình dự đoán đúng.")
        else:
            st.warning(f"Nhãn đúng là {label}.")
        st.write(
            "Kernel tích chập đã học:",
            model.convolution_weights.detach().cpu().numpy(),
        )

with q6_tab:
    st.subheader("Q6 — Scaled dot-product attention")
    st.write(
        "Nhập embedding cho từng token. Mô hình tính Q, K, V, áp dụng causal "
        "mask để mỗi vị trí chỉ chú ý đến chính nó và các vị trí trước đó."
    )
    token_text = st.text_input(
        "Các token theo thứ tự, phân cách bằng dấu phẩy",
        value="A, B, C",
    )
    tokens = [token.strip() for token in token_text.split(",") if token.strip()]
    default_vectors = [
        "1, 0, 0, 1",
        "0, 1, 1, 0",
        "1, 1, 0, 0",
    ]
    default_embeddings = "\n".join(
        default_vectors[index % len(default_vectors)]
        for index in range(len(tokens))
    )
    embedding_text = st.text_area(
        "Embedding — mỗi token một dòng, 4 số phân cách bằng dấu phẩy",
        value=default_embeddings,
        height=120,
        key=f"q6_embeddings_{len(tokens)}",
    )
    if not tokens:
        st.warning("Nhập ít nhất một token.")
    elif len(tokens) > 8:
        st.warning("Mô phỏng hỗ trợ tối đa 8 token.")
    else:
        try:
            embeddings = parse_embedding_matrix(embedding_text, len(tokens))
        except ValueError as error:
            st.warning(str(error))
        else:
            model = create_q6_model()
            with torch.no_grad():
                queries = model.q_layer(embeddings)
                keys = model.k_layer(embeddings)
                values = model.v_layer(embeddings)
                scores = queries @ keys.T / (model.layer_size ** 0.5)
                causal_mask = torch.tril(torch.ones_like(scores, dtype=torch.bool))
                masked_scores = scores.masked_fill(~causal_mask, float("-inf"))
                weights = torch.softmax(masked_scores, dim=-1)
                output = model(embeddings.unsqueeze(0))[0]

            matrix_column, output_column = st.columns(2)
            attention_figure, attention_axis = plt.subplots()
            attention_axis.imshow(weights.cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
            attention_axis.set(
                xticks=range(len(tokens)),
                yticks=range(len(tokens)),
                xticklabels=tokens,
                yticklabels=tokens,
                xlabel="Key — vị trí được chú ý",
                ylabel="Query — vị trí hiện tại",
                title="Trọng số attention (causal mask)",
            )
            for row in range(len(tokens)):
                for column in range(len(tokens)):
                    attention_axis.text(
                        column,
                        row,
                        f"{weights[row, column].item():.2f}",
                        ha="center",
                        va="center",
                    )
            matrix_column.pyplot(attention_figure)
            plt.close(attention_figure)
            output_column.write("Embedding đầu ra sau attention:")
            output_column.dataframe(
                {
                    "Token": tokens,
                    **{
                        f"dim {index + 1}": output[:, index].cpu().tolist()
                        for index in range(output.shape[1])
                    },
                },
                hide_index=True,
            )
            st.caption(
                "Các ô phía trên đường chéo chính bị causal mask chặn; "
                "mỗi hàng trọng số còn lại có tổng bằng 1."
            )

with info_tab:
    st.subheader("Nội dung mô phỏng")
    st.markdown(
        "- **Q1:** Perceptron phân loại nhị phân trên các điểm 2D.\n"
        "- **Q2:** Mạng neural xấp xỉ hàm sin trong khoảng [−2π, 2π].\n"
        "- **Q3:** Mạng neural phân loại chữ số viết tay MNIST.\n"
        "- **Q4:** RNN nhận diện ngôn ngữ từ một từ.\n"
        "- **Q5:** CNN phân loại chữ số và trực quan hóa feature map.\n"
        "- **Q6:** Scaled dot-product attention với causal mask."
    )
