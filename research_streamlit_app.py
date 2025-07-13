import streamlit as st
import json
import logging
import requests
from datetime import datetime
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# AI Provider Configuration
AI_PROVIDERS = {
    'google_gemini': {
        'api_key': 'AIzaSyCRhyQFcV5BzgKcxGrB_3m_tIdy85yDkk8',  # Replace with your actual API key
        'url': 'https://generativelanguage.googleapis.com/v1beta/models/',
        'models': [
            'gemini-2.0-flash',
            'gemini-1.5-flash-latest',
            'gemini-pro',
        ]
    }
}

# YouTube Data API Configuration
YOUTUBE_API_KEY = 'AIzaSyCRhyQFcV5BzgKcxGrB_3m_tIdy85yDkk8'  # Replace with your actual API key

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def create_research_prompt(topic: str, academic_level: str = "Graduate", research_area: str = "", 
                          keywords: str = "", user_context: str = "", word_count: int = 2000) -> str:
    """Create comprehensive research prompt"""
    
    prompt = f"""
    You are an advanced research assistant. Generate comprehensive research content for: "{topic}"
    
    Context:
    - Academic Level: {academic_level}
    - Research Area: {research_area}
    - Keywords: {keywords}
    - User: {user_context}
    - Required Word Count: {word_count} words
    
    Return ONLY a valid JSON object with these exact keys:
    
    {{
        "content": "Detailed research summary ({word_count}+ words) covering current research state, key theories, methodologies, recent developments, research gaps, applications, and conclusions. Make it comprehensive and academic.",
        
        "video_search_queries": [
            "Relevant Youtube query 1",
            "Relevant Youtube query 2",
            "Relevant Youtube query 3",
            "Relevant Youtube query 4",
            "Relevant Youtube query 5"
        ],
        
        "documents": [
            {{
                "title": "Document Title",
                "authors": "Author names",
                "source": "Journal/Publisher",
                "year": "Year",
                "url": "URL or DOI",
                "description": "Document description",
                "type": "research_paper/report/thesis",
                "relevance": "Research relevance"
            }}
        ],
        
        "links": [
            {{
                "title": "Resource Title",
                "url": "https://example.com",
                "description": "Resource description", 
                "type": "database/organization/tool",
                "relevance": "Research utility"
            }}
        ],
        
        "linkedin_profiles": [
            {{
                "name": "Expert Name",
                "title": "Position/Title",
                "institution": "Organization",
                "linkedin_url": "https://linkedin.com/in/profile",
                "expertise": "Expertise areas",
                "background": "Professional background",
                "relevance": "Why relevant for research",
                "contact_potential": "High/Medium/Low"
            }}
        ]
    }}
    
    Requirements:
    - Provide exactly 8-10 items for each array (documents, links)
    - Provide exactly 6-8 LinkedIn profiles
    - Provide 3-5 highly relevant video_search_queries for YouTube
    - Use real, accessible URLs for documents and links
    - Ensure all content is appropriate for {academic_level} level
    - Content should be exactly {word_count} words or more
    - Return ONLY the JSON object, no additional text
    """
    
    return prompt

def get_gemini_response(prompt: str, model: str = None) -> Tuple[Optional[str], str]:
    """Get response from Google Gemini"""
    try:
        model = model or AI_PROVIDERS['google_gemini']['models'][0]
        api_key = AI_PROVIDERS['google_gemini']['api_key']
        url = f"{AI_PROVIDERS['google_gemini']['url']}{model}:generateContent?key={api_key}"

        headers = {'Content-Type': 'application/json'}
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 8192,
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        response_data = response.json()

        if 'candidates' in response_data and response_data['candidates']:
            content = response_data['candidates'][0]['content']['parts'][0]['text']
            return content.strip(), model
        else:
            logger.error(f"Gemini unexpected response format: {response_data}")
            return None, model
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Gemini request error with model {model}: {e}")
        return None, model
    except Exception as e:
        logger.error(f"Google Gemini error with model {model}: {e}")
        return None, model

def fetch_youtube_videos(query: str, max_results: int = 5) -> List[Dict]:
    """Fetch YouTube videos using the YouTube Data API"""
    if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY':
        logger.error("YouTube API key is not configured.")
        return []

    logger.info(f"Fetching YouTube videos for query: '{query}' with max_results: {max_results}")
    
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results,
        "videoDuration": "long",
        "relevanceLanguage": "en"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        videos = []
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            
            videos.append({
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": snippet["thumbnails"]["high"]["url"] if "high" in snippet["thumbnails"] else "",
                "description": snippet["description"][:200] + "..." if len(snippet["description"]) > 200 else snippet["description"],
                "published": snippet["publishedAt"][:10],
                "relevance": f"Related to: {query}"
            })
        
        return videos
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching YouTube videos for query '{query}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in fetch_youtube_videos: {e}")
        return []

def parse_json_response(response_text: str) -> Dict:
    """Parse and validate AI response"""
    try:
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_text = response_text[json_start:json_end]
            parsed_data = json.loads(json_text)
            
            required_keys = ['content', 'video_search_queries', 'documents', 'links', 'linkedin_profiles']
            for key in required_keys:
                if key not in parsed_data:
                    parsed_data[key] = [] if key != 'content' else ""
            
            return parsed_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
    
    return {
        "content": "Unable to generate research content. Please check your API keys and try again.",
        "video_search_queries": [], 
        "documents": [], 
        "links": [], 
        "linkedin_profiles": []
    }

def generate_research_content(topic: str, academic_level: str, research_area: str, 
                            keywords: str, word_count: int) -> Dict:
    """Generate comprehensive research content"""
    try:
        user_context = f"A {academic_level} student researching {research_area}"
        research_prompt = create_research_prompt(topic, academic_level, research_area, keywords, user_context, word_count)
        
        with st.spinner("🤖 Generating research content with AI..."):
            ai_response, model_used = get_gemini_response(research_prompt)
        
        if not ai_response:
            return {"success": False, "error": "Failed to generate content from AI", "data": None}
        
        research_data = parse_json_response(ai_response)
        
        # Fetch YouTube videos
        all_youtube_videos = []
        if research_data.get('video_search_queries'):
            with st.spinner("📹 Fetching relevant YouTube videos..."):
                for query in research_data['video_search_queries']:
                    videos = fetch_youtube_videos(query, max_results=3)
                    all_youtube_videos.extend(videos)
                    time.sleep(0.5)  # Rate limiting
            
            research_data['videos'] = all_youtube_videos[:15]  # Top 15 videos
        
        return {
            "success": True,
            "error": None,
            "data": {
                "topic": topic,
                "academic_level": academic_level,
                "research_area": research_area,
                "keywords": keywords,
                "word_count": word_count,
                "research_content": research_data,
                "metadata": {
                    "ai_model": model_used,
                    "generated_at": datetime.now().isoformat(),
                    "total_videos": len(all_youtube_videos),
                    "videos_included": len(research_data.get('videos', [])),
                    "documents_found": len(research_data.get('documents', [])),
                    "links_found": len(research_data.get('links', [])),
                    "linkedin_profiles_found": len(research_data.get('linkedin_profiles', []))
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in generate_research_content: {e}")
        return {"success": False, "error": f"Internal error: {str(e)}", "data": None}

# --- Streamlit UI ---
def main():
    st.title("🔬 Research Assistant for PhD Students")
    st.markdown("Generate comprehensive research content with videos, documents, and expert profiles")
    
    # API Key Configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        gemini_key = st.text_input("Google Gemini API Key", 
                                 value=AI_PROVIDERS['google_gemini']['api_key'], 
                                 type="password",
                                 help="Get your API key from Google AI Studio")
        
        youtube_key = st.text_input("YouTube Data API Key", 
                                  value=YOUTUBE_API_KEY, 
                                  type="password",
                                  help="Get your API key from Google Cloud Console")
        
        # Update global variables
        AI_PROVIDERS['google_gemini']['api_key'] = gemini_key
        # global YOUTUBE_API_KEY
        # YOUTUBE_API_KEY = youtube_key
        
        st.divider()
        st.header("🎯 Research Settings")
        
        # Model selection
        selected_model = st.selectbox("Select AI Model", 
                                    AI_PROVIDERS['google_gemini']['models'],
                                    index=0)
        
        # Word count slider
        word_count = st.slider("Content Word Count", 
                             min_value=1000, 
                             max_value=10000, 
                             value=2000, 
                             step=500,
                             help="Number of words for the research content")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Research Query")
        
        # Research input form
        with st.form("research_form"):
            research_topic = st.text_input("Research Topic", 
                                         placeholder="e.g., Solar Energy Applications in Smart Cities",
                                         help="Enter your main research topic")
            
            academic_level = st.selectbox("Academic Level", 
                                        ["PhD", "Master's", "Graduate", "Undergraduate"],
                                        index=0)
            
            research_area = st.text_input("Research Area/Field", 
                                        placeholder="e.g., Renewable Energy, Environmental Science",
                                        help="Your specific field of study")
            
            keywords = st.text_input("Keywords", 
                                   placeholder="e.g., photovoltaic, sustainable energy, efficiency",
                                   help="Relevant keywords for your research")
            
            submit_button = st.form_submit_button("🚀 Generate Research Content", 
                                                type="primary",
                                                use_container_width=True)
    
    with col2:
        st.header("📊 Quick Stats")
        if 'research_result' in st.session_state:
            result = st.session_state.research_result
            if result['success']:
                metadata = result['data']['metadata']
                st.metric("📄 Documents Found", metadata['documents_found'])
                st.metric("🔗 Links Found", metadata['links_found'])
                st.metric("📹 Videos Found", metadata['videos_included'])
                st.metric("👥 LinkedIn Profiles", metadata['linkedin_profiles_found'])
        else:
            st.info("Submit a research query to see statistics")
    
    # Process form submission
    if submit_button:
        if not research_topic.strip():
            st.error("Please enter a research topic")
            return
        
        if not gemini_key or gemini_key == 'YOUR_GEMINI_API_KEY':
            st.error("Please configure your Google Gemini API key in the sidebar")
            return
        
        # Generate research content
        result = generate_research_content(
            topic=research_topic,
            academic_level=academic_level,
            research_area=research_area,
            keywords=keywords,
            word_count=word_count
        )
        
        st.session_state.research_result = result
    
    # Display results
    if 'research_result' in st.session_state:
        result = st.session_state.research_result
        
        if result['success']:
            data = result['data']
            research_content = data['research_content']
            
            st.success(f"✅ Research content generated successfully using {data['metadata']['ai_model']}")
            
            # Tabs for different content types
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Research Content", "📹 YouTube Videos", "📚 Documents", "🔗 Links", "👥 LinkedIn Profiles"])
            
            with tab1:
                st.header("📄 Research Content")
                st.markdown(f"**Topic:** {data['topic']}")
                st.markdown(f"**Academic Level:** {data['academic_level']}")
                st.markdown(f"**Research Area:** {data['research_area']}")
                st.markdown(f"**Keywords:** {data['keywords']}")
                st.divider()
                
                content = research_content.get('content', '')
                st.markdown(content)
                
                # Download button for content
                st.download_button(
                    label="📥 Download Research Content",
                    data=content,
                    file_name=f"research_{data['topic'].replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.header("📹 YouTube Videos")
                videos = research_content.get('videos', [])
                
                if videos:
                    for idx, video in enumerate(videos, 1):
                        with st.expander(f"Video {idx}: {video['title'][:50]}..."):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if video.get('thumbnail'):
                                    st.image(video['thumbnail'], width=200)
                            
                            with col2:
                                st.markdown(f"**Title:** {video['title']}")
                                st.markdown(f"**Channel:** {video['channel']}")
                                st.markdown(f"**Published:** {video.get('published', 'N/A')}")
                                st.markdown(f"**URL:** [{video['url']}]({video['url']})")
                                st.markdown(f"**Description:** {video['description']}")
                else:
                    st.info("No YouTube videos found. Please check your YouTube API key.")
            
            with tab3:
                st.header("📚 Research Documents")
                documents = research_content.get('documents', [])
                
                if documents:
                    for idx, doc in enumerate(documents, 1):
                        with st.expander(f"Document {idx}: {doc['title']}"):
                            st.markdown(f"**Authors:** {doc['authors']}")
                            st.markdown(f"**Source:** {doc['source']}")
                            st.markdown(f"**Year:** {doc['year']}")
                            st.markdown(f"**Type:** {doc['type']}")
                            st.markdown(f"**URL:** [{doc['url']}]({doc['url']})")
                            st.markdown(f"**Description:** {doc['description']}")
                            st.markdown(f"**Relevance:** {doc['relevance']}")
                else:
                    st.info("No documents found in the research results.")
            
            with tab4:
                st.header("🔗 Useful Links")
                links = research_content.get('links', [])
                
                if links:
                    for idx, link in enumerate(links, 1):
                        with st.expander(f"Link {idx}: {link['title']}"):
                            st.markdown(f"**Type:** {link['type']}")
                            st.markdown(f"**URL:** [{link['url']}]({link['url']})")
                            st.markdown(f"**Description:** {link['description']}")
                            st.markdown(f"**Relevance:** {link['relevance']}")
                else:
                    st.info("No links found in the research results.")
            
            with tab5:
                st.header("👥 LinkedIn Profiles")
                profiles = research_content.get('linkedin_profiles', [])
                
                if profiles:
                    for idx, profile in enumerate(profiles, 1):
                        with st.expander(f"Expert {idx}: {profile['name']}"):
                            st.markdown(f"**Title:** {profile['title']}")
                            st.markdown(f"**Institution:** {profile['institution']}")
                            st.markdown(f"**LinkedIn:** [{profile['linkedin_url']}]({profile['linkedin_url']})")
                            st.markdown(f"**Expertise:** {profile['expertise']}")
                            st.markdown(f"**Background:** {profile['background']}")
                            st.markdown(f"**Relevance:** {profile['relevance']}")
                            st.markdown(f"**Contact Potential:** {profile['contact_potential']}")
                else:
                    st.info("No LinkedIn profiles found in the research results.")
        
        else:
            st.error(f"❌ Error: {result['error']}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built for PhD students and researchers • Configure your API keys in the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
