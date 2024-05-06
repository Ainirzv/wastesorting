
# Load model
model = load_model(r'C:\Users\HP\Desktop\model_5class_resnet_87%.h5')

labels = ['G&M', 'Organic', 'Other', 'Paper', 'Plastic']
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

def preprocess_image(image):
    resized_img = cv2.resize(image, (224, 224))
    preprocessed_img = imagenet_utils.preprocess_input(resized_img)
    preprocessed_img = img_to_array(preprocessed_img)
    preprocessed_img = np.expand_dims(preprocessed_img / 255, 0)
    return preprocessed_img

def format_array_to_str(array):
    return str(array.tolist())[1:-1]

def main():
    st.title('Real-Time Object Classification')
    st.sidebar.title('Settings')
    
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2
    
    while True:
        start_time = time.time()
        success, img = cap.read()
        if not success:
            st.error("Failed to capture frame from webcam.")
            break

        preprocessed_img = preprocess_image(img)
        predictions = model.predict(preprocessed_img)
        confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions)

        fps = 1.0 / (time.time() - start_time)
        st.write('FPS:', fps)

        if confidence > 0.85:
            predicted_class = labels[predicted_class_index]
            cv2.putText(img, predicted_class, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(img, format_array_to_str(confidence), (50, 150), font, font_scale, color, thickness, cv2.LINE_AA)
            st.image(img, channels="BGR")
        
        if st.button('Stop'):
            break

if __name__ == '__main__':
    main()
