import streamlit as st
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import re
from urllib.parse import urlparse, urljoin
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Research Assistant for PhD Students",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .research-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    
    .video-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    .document-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .linkedin-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #0077b5;
        margin: 0.5rem 0;
    }
    
    .stTextArea textarea {
        min-height: 100px;
    }
</style>
""", unsafe_allow_html=True)

class ResearchAssistant:
    """Research Assistant class for generating comprehensive research content"""
    
    def __init__(self, gemini_api_key: str, youtube_api_key: str = None, serpapi_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.youtube_api_key = youtube_api_key
        self.serpapi_key = serpapi_key
        self.gemini_url = 'https://generativelanguage.googleapis.com/v1beta/models/'
        self.gemini_models = [
            'gemini-2.0-flash',
            'gemini-1.5-flash-latest',
            'gemini-pro',
        ]
    
    def verify_url(self, url: str) -> bool:
        """Verify if a URL is accessible"""
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    def search_web_content(self, query: str, search_type: str = "documents") -> List[Dict]:
        """Search for real web content using SerpAPI"""
        if not self.serpapi_key:
            return []
        
        try:
            # Modify query based on search type
            if search_type == "documents":
                query += " filetype:pdf OR site:arxiv.org OR site:researchgate.net OR site:scholar.google.com"
            elif search_type == "profiles":
                query += " site:linkedin.com/in"
            elif search_type == "academic":
                query += " site:edu OR site:org research"
            
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "engine": "google",
                "api_key": self.serpapi_key,
                "num": 10
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic_results", []):
                if self.verify_url(item.get("link", "")):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "description": item.get("snippet", ""),
                        "source": urlparse(item.get("link", "")).netloc
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    def create_research_prompt(self, topic: str, academic_level: str = "PhD", 
                              research_area: str = "", keywords: str = "", 
                              user_context: str = "", word_count: int = 2000) -> str:
        """Create comprehensive research prompt for AI"""
        prompt = f"""
        You are an advanced research assistant. Generate comprehensive research content for: "{topic}"
        
        Context:
        - Academic Level: {academic_level}
        - Research Area: {research_area}
        - Keywords: {keywords}
        - User: {user_context}
        - Required Word Count: {word_count} words
        
        IMPORTANT: Do NOT generate fake URLs or links. Only provide titles, descriptions, and search terms.
        
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
            
            "document_search_queries": [
                "Academic paper search query 1",
                "Academic paper search query 2",
                "Academic paper search query 3",
                "Research document query 4",
                "Thesis dissertation query 5"
            ],
            
            "web_search_queries": [
                "Organization database query 1",
                "Research tool query 2",
                "Educational resource query 3",
                "Professional resource query 4",
                "Industry report query 5"
            ],
            
            "linkedin_search_queries": [
                "Expert researcher query 1",
                "Academic professional query 2",
                "Industry specialist query 3",
                "Research scientist query 4",
                "Professor specialist query 5"
            ],
            
            "suggested_experts": [
                {{
                    "name": "Well-known expert name in the field",
                    "title": "Typical position/title",
                    "institution": "Typical organization type",
                    "expertise": "Expertise areas",
                    "background": "Professional background",
                    "relevance": "Why relevant for research",
                    "search_terms": "Name + field + institution search terms"
                }}
            ],
            
            "suggested_sources": [
                {{
                    "title": "Typical academic source title",
                    "authors": "Typical author format",
                    "source": "Journal/Publisher type",
                    "year": "Recent year",
                    "description": "Document description",
                    "type": "research_paper/report/thesis",
                    "relevance": "Research relevance",
                    "search_terms": "Title + authors + keywords search terms"
                }}
            ]
        }}
        
        Requirements:
        - Provide exactly 5 search queries for each query array
        - Provide exactly 6-8 suggested experts and sources
        - Provide 3-5 highly relevant video_search_queries for YouTube
        - Use realistic but generic information (no specific URLs)
        - Ensure all content is appropriate for {academic_level} level
        - Content should be exactly {word_count} words or more
        - Return ONLY the JSON object, no additional text
        """
        return prompt
    
    def get_gemini_response(self, prompt: str, model: str = None) -> Tuple[Optional[str], str]:
        """Get response from Google Gemini API"""
        try:
            model = model or self.gemini_models[0]
            url = f"{self.gemini_url}{model}:generateContent?key={self.gemini_api_key}"

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
                return None, model
                
        except Exception as e:
            logger.error(f"Google Gemini error with model {model}: {e}")
            return None, model
    
    def fetch_youtube_videos(self, query: str, max_results: int = 5) -> List[Dict]:
        """Fetch YouTube videos using YouTube Data API"""
        if not self.youtube_api_key:
            return []

        base_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "key": self.youtube_api_key,
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
            
        except Exception as e:
            logger.error(f"Error fetching YouTube videos: {e}")
            return []
    
    def parse_json_response(self, response_text: str) -> Dict:
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
                
                required_keys = ['content', 'video_search_queries', 'document_search_queries', 
                               'web_search_queries', 'linkedin_search_queries', 'suggested_experts', 'suggested_sources']
                for key in required_keys:
                    if key not in parsed_data:
                        parsed_data[key] = [] if key != 'content' else ""
                
                return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
        
        return {
            "content": "Unable to generate research content. Please check your API keys and try again.",
            "video_search_queries": [], 
            "document_search_queries": [],
            "web_search_queries": [],
            "linkedin_search_queries": [],
            "suggested_experts": [],
            "suggested_sources": []
        }
    
    def generate_research_content(self, topic: str, academic_level: str = "PhD", 
                                 research_area: str = "", keywords: str = "", 
                                 word_count: int = 2000, include_videos: bool = True,
                                 include_web_search: bool = True) -> Dict:
        """Generate comprehensive research content with real web search"""
        try:
            user_context = f"A {academic_level} student researching {research_area}"
            research_prompt = self.create_research_prompt(
                topic, academic_level, research_area, keywords, user_context, word_count
            )
            
            # Get AI response
            ai_response, model_used = self.get_gemini_response(research_prompt)
            
            if not ai_response:
                return {
                    "success": False, 
                    "error": "Failed to generate content from AI", 
                    "data": None
                }
            
            # Parse response
            research_data = self.parse_json_response(ai_response)
            
            # Fetch YouTube videos if requested
            all_youtube_videos = []
            if include_videos and self.youtube_api_key and research_data.get('video_search_queries'):
                for query in research_data['video_search_queries']:
                    videos = self.fetch_youtube_videos(query, max_results=3)
                    all_youtube_videos.extend(videos)
                    time.sleep(0.5)  # Rate limiting
                
                research_data['videos'] = all_youtube_videos[:15]  # Top 15 videos
            
            # Perform web search for real content if requested
            real_documents = []
            real_links = []
            real_linkedin_profiles = []
            
            if include_web_search and self.serpapi_key:
                # Search for documents
                for query in research_data.get('document_search_queries', []):
                    doc_results = self.search_web_content(query, "documents")
                    for result in doc_results:
                        real_documents.append({
                            "title": result["title"],
                            "url": result["url"],
                            "description": result["description"],
                            "source": result["source"],
                            "type": "research_paper" if "pdf" in result["url"] else "webpage",
                            "relevance": f"Found via search: {query}"
                        })
                
                # Search for general links
                for query in research_data.get('web_search_queries', []):
                    web_results = self.search_web_content(query, "general")
                    for result in web_results:
                        real_links.append({
                            "title": result["title"],
                            "url": result["url"],
                            "description": result["description"],
                            "source": result["source"],
                            "type": "resource",
                            "relevance": f"Found via search: {query}"
                        })
                
                # Search for LinkedIn profiles
                for query in research_data.get('linkedin_search_queries', []):
                    profile_results = self.search_web_content(query, "profiles")
                    for result in profile_results:
                        real_linkedin_profiles.append({
                            "name": result["title"].replace(" | LinkedIn", "").replace(" - LinkedIn", ""),
                            "linkedin_url": result["url"],
                            "description": result["description"],
                            "relevance": f"Found via search: {query}",
                            "contact_potential": "Medium"
                        })
            
            # Use real search results if available, otherwise use suggestions
            final_documents = real_documents if real_documents else research_data.get('suggested_sources', [])
            final_links = real_links if real_links else []
            final_linkedin_profiles = real_linkedin_profiles if real_linkedin_profiles else research_data.get('suggested_experts', [])
            
            return {
                "success": True,
                "error": None,
                "data": {
                    "topic": topic,
                    "academic_level": academic_level,
                    "research_area": research_area,
                    "keywords": keywords,
                    "word_count": word_count,
                    "content": research_data.get('content', ''),
                    "videos": research_data.get('videos', []),
                    "documents": final_documents,
                    "links": final_links,
                    "linkedin_profiles": final_linkedin_profiles,
                    "search_queries": {
                        "videos": research_data.get('video_search_queries', []),
                        "documents": research_data.get('document_search_queries', []),
                        "web": research_data.get('web_search_queries', []),
                        "linkedin": research_data.get('linkedin_search_queries', [])
                    },
                    "metadata": {
                        "ai_model": model_used,
                        "generated_at": datetime.now().isoformat(),
                        "total_videos": len(all_youtube_videos),
                        "videos_included": len(research_data.get('videos', [])),
                        "documents_found": len(final_documents),
                        "links_found": len(final_links),
                        "linkedin_profiles_found": len(final_linkedin_profiles),
                        "web_search_enabled": include_web_search and self.serpapi_key is not None
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in generate_research_content: {e}")
            return {
                "success": False, 
                "error": f"Internal error: {str(e)}", 
                "data": None
            }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Research Assistant for PhD Students</h1>
        <p>Generate comprehensive research content, find academic resources, and discover relevant experts</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API configuration
    st.sidebar.header("🔑 API Configuration")
    
    with st.sidebar.expander("API Keys Setup", expanded=True):
        gemini_api_key = st.text_input("Gemini API Key", type="password", 
                                      help="Required for AI content generation")
        youtube_api_key = st.text_input("YouTube API Key", type="password", 
                                       help="Optional - for YouTube video search")
        serpapi_key = st.text_input("SerpAPI Key", type="password", 
                                   help="Optional - for real-time web search")
    
    # Research Configuration
    st.sidebar.header("📚 Research Settings")
    
    academic_level = st.sidebar.selectbox(
        "Academic Level",
        ["PhD", "Master's", "Bachelor's", "Postdoc", "Faculty"]
    )
    
    word_count = st.sidebar.slider(
        "Content Word Count",
        min_value=1000,
        max_value=5000,
        value=2000,
        step=500
    )
    
    include_videos = st.sidebar.checkbox("Include YouTube Videos", value=True)
    include_web_search = st.sidebar.checkbox("Include Web Search", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Research Topic Input")
        
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., AI and Machine Learning Role in Social Development",
            help="Enter your main research topic"
        )
        
        research_area = st.text_input(
            "Research Area",
            placeholder="e.g., Artificial Intelligence, Computer Science",
            help="Specific field or domain of research"
        )
        
        keywords = st.text_area(
            "Keywords",
            placeholder="e.g., machine learning, social impact, technology adoption, digital transformation",
            help="Relevant keywords separated by commas"
        )
    
    with col2:
        st.header("🎯 Quick Actions")
        
        if st.button("🔍 Generate Research", type="primary", use_container_width=True):
            if not topic.strip():
                st.error("Please enter a research topic")
                return
            
            if not gemini_api_key:
                st.error("Please provide a Gemini API key")
                return
            
            # Show progress
            with st.spinner("Generating research content..."):
                progress_bar = st.progress(0)
                
                # Initialize research assistant
                assistant = ResearchAssistant(
                    gemini_api_key=gemini_api_key,
                    youtube_api_key=youtube_api_key,
                    serpapi_key=serpapi_key
                )
                
                progress_bar.progress(25)
                
                # Generate research content
                result = assistant.generate_research_content(
                    topic=topic,
                    academic_level=academic_level,
                    research_area=research_area,
                    keywords=keywords,
                    word_count=word_count,
                    include_videos=include_videos,
                    include_web_search=include_web_search
                )
                
                progress_bar.progress(100)
                
                if result["success"]:
                    st.session_state.research_data = result["data"]
                    st.success("Research content generated successfully!")
                else:
                    st.error(f"Error: {result['error']}")
        
        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            if 'research_data' in st.session_state:
                del st.session_state.research_data
            st.rerun()
    
    # Display results if available
    if 'research_data' in st.session_state:
        data = st.session_state.research_data
        
        st.markdown("---")
        st.header("📊 Research Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(data.get('videos', []))}</h3>
                <p>YouTube Videos</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(data.get('documents', []))}</h3>
                <p>Research Papers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(data.get('links', []))}</h3>
                <p>Web Resources</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(data.get('linkedin_profiles', []))}</h3>
                <p>Expert Profiles</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main content
        st.markdown("### 📄 Research Content")
        with st.expander("View Full Research Content", expanded=True):
            st.markdown(f"""
            <div class="research-card">
                {data.get('content', 'No content available')}
            </div>
            """, unsafe_allow_html=True)
        
        # Create tabs for different types of resources
        tab1, tab2, tab3, tab4 = st.tabs(["📹 Videos", "📚 Documents", "🔗 Web Resources", "👥 Experts"])
        
        with tab1:
            st.markdown("### YouTube Videos")
            videos = data.get('videos', [])
            if videos:
                for video in videos:
                    st.markdown(f"""
                    <div class="video-card">
                        <h4><a href="{video['url']}" target="_blank">{video['title']}</a></h4>
                        <p><strong>Channel:</strong> {video['channel']}</p>
                        <p><strong>Published:</strong> {video['published']}</p>
                        <p>{video['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No YouTube videos found. Check your YouTube API key.")
        
        with tab2:
            st.markdown("### Research Documents & Papers")
            documents = data.get('documents', [])
            if documents:
                for doc in documents:
                    st.markdown(f"""
                    <div class="document-card">
                        <h4><a href="{doc.get('url', '#')}" target="_blank">{doc.get('title', 'No title')}</a></h4>
                        <p><strong>Source:</strong> {doc.get('source', 'Unknown')}</p>
                        <p><strong>Type:</strong> {doc.get('type', 'Document')}</p>
                        <p>{doc.get('description', 'No description available')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No research documents found. This may be due to API limitations or search settings.")
        
        with tab3:
            st.markdown("### Web Resources")
            links = data.get('links', [])
            if links:
                for link in links:
                    st.markdown(f"""
                    <div class="document-card">
                        <h4><a href="{link.get('url', '#')}" target="_blank">{link.get('title', 'No title')}</a></h4>
                        <p><strong>Source:</strong> {link.get('source', 'Unknown')}</p>
                        <p>{link.get('description', 'No description available')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No web resources found. Enable web search and provide SerpAPI key for better results.")
        
        with tab4:
            st.markdown("### Expert Profiles")
            profiles = data.get('linkedin_profiles', [])
            if profiles:
                for profile in profiles:
                    st.markdown(f"""
                    <div class="linkedin-card">
                        <h4><a href="{profile.get('linkedin_url', '#')}" target="_blank">{profile.get('name', 'Unknown')}</a></h4>
                        <p>{profile.get('description', 'No description available')}</p>
                        <p><strong>Relevance:</strong> {profile.get('relevance', 'Not specified')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No expert profiles found. Enable web search and provide SerpAPI key for LinkedIn profile search.")
        
        # Download options
        st.markdown("### 💾 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            if st.button("📄 Download as JSON"):
                json_data = json.dumps(data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"research_{topic.replace(' ', '_')}.json",
                    mime="application/json"
                )
        
        with col2:
            # Export as text summary
            if st.button("📝 Download Summary"):
                summary = f"""
Research Topic: {data.get('topic', 'N/A')}
Academic Level: {data.get('academic_level', 'N/A')}
Research Area: {data.get('research_area', 'N/A')}
Keywords: {data.get('keywords', 'N/A')}
Generated: {data.get('metadata', {}).get('generated_at', 'N/A')}

RESEARCH CONTENT:
{data.get('content', 'No content available')}

RESOURCES FOUND:
- Videos: {len(data.get('videos', []))}
- Documents: {len(data.get('documents', []))}
- Web Resources: {len(data.get('links', []))}
- Expert Profiles: {len(data.get('linkedin_profiles', []))}
                """
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"research_summary_{topic.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🔬 Research Assistant for PhD Students | Built with Streamlit</p>
        <p>Configure your API keys in the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
