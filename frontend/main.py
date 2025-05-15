# frontend/main.py
import streamlit as st
import requests
import json

def main():
    st.title("Fuse Machine")
    st.title("Text-to-Image Search: ResNet vs ViT Comparison")
    st.write("Enter a text query to compare search results and similarity scores from ResNet and ViT models.")

    query = st.text_input("Query", value="working on laptop in office")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

    if st.button("Search"):
        query = query.strip()  # Trim whitespace to avoid empty queries
        if not query:
            st.error("Please enter a valid query.")
            return

        payload = {"query": query, "top_k": top_k}
        st.write(f"Sending payload: {payload}")  # Debug: Log payload to verify

        with st.spinner("Searching..."):
            try:
                # Call backend API
                response = requests.post(
                    "http://backend:8000/search",
                    json=payload
                )
                response.raise_for_status()
                results = response.json()

                resnet_results = results.get("resnet", [])
                vit_results = results.get("vit", [])

                if resnet_results or vit_results:
                    st.subheader("Search Results")
                    for i in range(max(len(resnet_results), len(vit_results))):
                        col1, col2 = st.columns(2)
                        with col1:
                            if i < len(resnet_results):
                                image_data = resnet_results[i]
                                st.image(
                                    image_data["image"],
                                    caption=f"ResNet: {image_data['path']}, Similarity: {image_data['similarity']:.4f}",
                                    use_container_width=True
                                )
                            else:
                                st.write("No more results.")
                        with col2:
                            if i < len(vit_results):
                                image_data = vit_results[i]
                                st.image(
                                    image_data["image"],
                                    caption=f"ViT: {image_data['path']}, Similarity: {image_data['similarity']:.4f}",
                                    use_container_width=True
                                )
                            else:
                                st.write("No more results.")

                # Placeholder for metrics (to be implemented in backend)
                st.subheader("Metrics Comparison")
                st.write("Metrics to be implemented in backend API.")

                st.success(f"Found {len(resnet_results)} ResNet results and {len(vit_results)} ViT results for query: '{query}'")

            except requests.exceptions.HTTPError as e:
                if response.status_code == 422:
                    st.error(f"Invalid request: {json.dumps(response.json()['detail'], indent=2)}")
                else:
                    st.error(f"Error during search: {str(e)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main()