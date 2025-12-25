"""
Test script to verify Streamlit audio_input behavior
"""
import streamlit as st
import base64

st.title("Debug Audio Input")

audio = st.audio_input("Test recorder")

if audio:
    st.write(f"Type of audio: {type(audio)}")
    st.write(f"Audio name: {audio.name}")
    
    # Try reading
    try:
        data = audio.read()
        st.write(f"Type of data after read(): {type(data)}")
        st.write(f"First 10 bytes: {data[:10]}")
        
        # Try encoding
        try:
            encoded = base64.b64encode(data)
            st.write(f"Encoded type: {type(encoded)}")
            
            decoded_str = encoded.decode("ascii")
            st.write(f"Decoded string type: {type(decoded_str)}")
            st.write(f"First 20 chars: {decoded_str[:20]}")
            
            st.success("✅ Encoding successful!")
        except Exception as e:
            st.error(f"❌ Encoding error: {e}")
            
    except Exception as e:
        st.error(f"❌ Read error: {e}")
