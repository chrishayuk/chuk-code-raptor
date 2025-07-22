#!/usr/bin/env python3
"""
RSS/Atom Parser Demo Script
============================

Comprehensive demonstration of the RSS/Atom parser capabilities including:
- RSS 2.0, RSS 1.0, and Atom 1.0 feed parsing
- Feed metadata extraction and semantic analysis
- Entry/item content processing with full-text extraction
- Media enclosure handling (podcasts, videos)
- Cross-feed comparison and analytics
- Content freshness and quality scoring
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_sample_feed_data() -> Dict[str, str]:
    """Create various RSS/Atom feed samples for testing"""
    
    samples = {}
    
    # RSS 2.0 Tech News Feed
    samples['rss2_tech_news'] = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:media="http://search.yahoo.com/mrss/">
    
    <channel>
        <title>TechNews Daily</title>
        <link>https://technews.example.com</link>
        <description>Latest technology news and insights from the digital world</description>
        <language>en-us</language>
        <copyright>© 2025 TechNews Daily</copyright>
        <managingEditor>editor@technews.example.com (Tech Editor)</managingEditor>
        <webMaster>webmaster@technews.example.com</webMaster>
        <pubDate>Tue, 15 Jan 2025 10:30:00 GMT</pubDate>
        <lastBuildDate>Tue, 15 Jan 2025 10:30:00 GMT</lastBuildDate>
        <category>Technology</category>
        <category>News</category>
        <generator>TechNews CMS v2.1</generator>
        <ttl>60</ttl>
        
        <image>
            <url>https://technews.example.com/logo.png</url>
            <title>TechNews Daily</title>
            <link>https://technews.example.com</link>
            <width>144</width>
            <height>144</height>
        </image>
        
        <item>
            <title>Breakthrough in Quantum Computing: 1000-Qubit Processor Announced</title>
            <link>https://technews.example.com/articles/quantum-breakthrough-2025</link>
            <description>A revolutionary 1000-qubit quantum processor has been unveiled, promising unprecedented computational capabilities for complex problem solving.</description>
            <content:encoded><![CDATA[
                <p>In a groundbreaking announcement that could reshape the landscape of computing, 
                QuantumTech Corp revealed their latest achievement: a fully functional 1000-qubit 
                quantum processor. This milestone represents a significant leap forward in quantum 
                computing capabilities.</p>
                
                <h3>Technical Specifications</h3>
                <ul>
                    <li>1000 stable qubits with 99.9% fidelity</li>
                    <li>Coherence time exceeding 100 microseconds</li>
                    <li>Gate operation time under 10 nanoseconds</li>
                    <li>Error correction using surface code topology</li>
                </ul>
                
                <p>The implications for fields such as cryptography, drug discovery, and 
                artificial intelligence are profound. Early benchmarks suggest the processor 
                can solve certain optimization problems exponentially faster than classical 
                supercomputers.</p>
                
                <blockquote>
                    "This represents the dawn of practical quantum advantage for real-world 
                    applications," said Dr. Sarah Chen, lead quantum architect at QuantumTech.
                </blockquote>
            ]]></content:encoded>
            <pubDate>Tue, 15 Jan 2025 09:00:00 GMT</pubDate>
            <author>sarah.chen@technews.example.com (Sarah Chen)</author>
            <category>Quantum Computing</category>
            <category>Hardware</category>
            <category>Research</category>
            <guid isPermaLink="true">https://technews.example.com/articles/quantum-breakthrough-2025</guid>
            <dc:creator>Sarah Chen</dc:creator>
            <dc:date>2025-01-15T09:00:00Z</dc:date>
            <dc:subject>Quantum Computing</dc:subject>
        </item>
        
        <item>
            <title>AI Safety Summit 2025: Global Leaders Establish New Guidelines</title>
            <link>https://technews.example.com/articles/ai-safety-summit-2025</link>
            <description>World leaders and AI researchers convene to establish comprehensive safety guidelines for artificial general intelligence development.</description>
            <content:encoded><![CDATA[
                <p>The AI Safety Summit 2025 concluded with unprecedented international 
                cooperation on artificial intelligence governance. Representatives from 
                50 countries, leading tech companies, and research institutions reached 
                consensus on critical safety measures.</p>
                
                <h3>Key Agreements</h3>
                <ol>
                    <li><strong>Mandatory Safety Testing:</strong> All AI systems above 
                    specified capability thresholds must undergo rigorous safety evaluation</li>
                    <li><strong>International Monitoring Body:</strong> Creation of a global 
                    AI safety organization with oversight powers</li>
                    <li><strong>Open Research Initiative:</strong> Shared database of AI 
                    safety research and incident reporting</li>
                    <li><strong>Ethical Guidelines:</strong> Binding principles for AI 
                    development and deployment</li>
                </ol>
                
                <p>The summit addressed growing concerns about AI alignment, transparency, 
                and the potential risks of advanced AI systems. Industry leaders praised 
                the collaborative approach while emphasizing the need for continued innovation.</p>
            ]]></content:encoded>
            <pubDate>Mon, 14 Jan 2025 15:30:00 GMT</pubDate>
            <author>michael.torres@technews.example.com (Michael Torres)</author>
            <category>Artificial Intelligence</category>
            <category>Policy</category>
            <category>Safety</category>
            <guid isPermaLink="true">https://technews.example.com/articles/ai-safety-summit-2025</guid>
            <dc:creator>Michael Torres</dc:creator>
        </item>
        
        <item>
            <title>Sustainable Tech: Solar Panel Efficiency Reaches 35% Milestone</title>
            <link>https://technews.example.com/articles/solar-efficiency-milestone</link>
            <description>New perovskite-silicon tandem solar cells achieve record-breaking 35% efficiency in laboratory conditions.</description>
            <content:encoded><![CDATA[
                <p>Researchers at SolarTech Institute have achieved a major breakthrough 
                in photovoltaic technology, developing solar cells with 35% efficiency 
                under standard test conditions. This represents a significant advancement 
                over current commercial panels that typically achieve 20-22% efficiency.</p>
                
                <div class="highlight-box">
                    <h4>Efficiency Comparison</h4>
                    <ul>
                        <li>New Technology: 35.2% efficiency</li>
                        <li>Current Best Commercial: 22.8% efficiency</li>
                        <li>Previous Lab Record: 33.7% efficiency</li>
                    </ul>
                </div>
                
                <p>The breakthrough combines perovskite and silicon technologies in a 
                tandem cell structure. The top perovskite layer captures high-energy 
                photons while the silicon bottom layer processes lower-energy light, 
                maximizing overall energy conversion.</p>
            ]]></content:encoded>
            <pubDate>Sun, 13 Jan 2025 12:00:00 GMT</pubDate>
            <author>elena.rodriguez@technews.example.com (Elena Rodriguez)</author>
            <category>Renewable Energy</category>
            <category>Solar</category>
            <category>Research</category>
            <guid isPermaLink="true">https://technews.example.com/articles/solar-efficiency-milestone</guid>
        </item>
    </channel>
</rss>"""

    # Atom 1.0 Science Blog Feed
    samples['atom_science_blog'] = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">
    
    <title>Science Insights Blog</title>
    <subtitle>Exploring the frontiers of scientific discovery</subtitle>
    <link href="https://scienceinsights.example.com/"/>
    <link rel="self" type="application/atom+xml" href="https://scienceinsights.example.com/feed.atom"/>
    <id>https://scienceinsights.example.com/</id>
    <updated>2025-01-15T14:22:00Z</updated>
    <rights>© 2025 Science Insights Blog. Creative Commons Attribution License.</rights>
    <generator uri="https://jekyllrb.com/" version="4.3.2">Jekyll</generator>
    
    <author>
        <name>Dr. Alex Johnson</name>
        <email>alex@scienceinsights.example.com</email>
        <uri>https://scienceinsights.example.com/about</uri>
    </author>
    
    <category term="science" label="Science"/>
    <category term="research" label="Research"/>
    <category term="discovery" label="Discovery"/>
    
    <entry>
        <title>CRISPR 3.0: The Next Generation of Gene Editing</title>
        <link href="https://scienceinsights.example.com/posts/crispr-3-next-generation"/>
        <id>https://scienceinsights.example.com/posts/crispr-3-next-generation</id>
        <updated>2025-01-15T14:22:00Z</updated>
        <published>2025-01-15T09:15:00Z</published>
        
        <author>
            <name>Dr. Lisa Park</name>
            <email>lisa.park@scienceinsights.example.com</email>
        </author>
        
        <category term="genetics" label="Genetics"/>
        <category term="biotechnology" label="Biotechnology"/>
        <category term="crispr" label="CRISPR"/>
        
        <summary type="text">Scientists unveil CRISPR 3.0, featuring enhanced precision, reduced off-target effects, and new therapeutic applications in rare genetic diseases.</summary>
        
        <content type="html"><![CDATA[
            <article>
                <h2>Revolutionary Precision in Gene Editing</h2>
                
                <p>The scientific community is buzzing with excitement over the latest 
                advancement in gene editing technology. CRISPR 3.0, developed through 
                collaboration between leading research institutions, represents a quantum 
                leap in precision and safety for genetic modifications.</p>
                
                <h3>Key Improvements</h3>
                
                <div class="improvement-grid">
                    <div class="improvement-item">
                        <h4>Enhanced Precision</h4>
                        <p>Off-target effects reduced by 99.8% compared to previous versions</p>
                    </div>
                    
                    <div class="improvement-item">
                        <h4>Expanded Targeting</h4>
                        <p>Can now target previously inaccessible genomic regions</p>
                    </div>
                    
                    <div class="improvement-item">
                        <h4>Improved Delivery</h4>
                        <p>Novel delivery mechanisms increase efficiency by 400%</p>
                    </div>
                </div>
                
                <h3>Clinical Applications</h3>
                
                <p>Early trials show promising results for treating:</p>
                <ul>
                    <li><strong>Huntington's Disease:</strong> 85% reduction in mutant protein expression</li>
                    <li><strong>Sickle Cell Anemia:</strong> Complete correction in 92% of treated cells</li>
                    <li><strong>Duchenne Muscular Dystrophy:</strong> Restored dystrophin expression</li>
                    <li><strong>Leber Congenital Amaurosis:</strong> Vision improvement in animal models</li>
                </ul>
                
                <blockquote>
                    <p>"CRISPR 3.0 brings us closer to the dream of personalized genetic medicine. 
                    We're not just editing genes anymore; we're precisely orchestrating cellular 
                    symphony," explained Dr. Maria Santos, lead researcher on the project.</p>
                </blockquote>
                
                <h3>Ethical Considerations</h3>
                
                <p>With great power comes great responsibility. The research team emphasizes 
                the importance of ethical guidelines and public engagement as this technology 
                advances toward clinical applications.</p>
                
                <div class="callout-box">
                    <h4>Looking Ahead</h4>
                    <p>Phase I clinical trials are expected to begin in Q3 2025, pending 
                    regulatory approval. The first applications will focus on severe genetic 
                    disorders with limited treatment options.</p>
                </div>
            </article>
        ]]></content>
    </entry>
    
    <entry>
        <title>Climate Modeling Breakthrough: AI Predicts Weather Patterns 30 Days Out</title>
        <link href="https://scienceinsights.example.com/posts/ai-weather-prediction-breakthrough"/>
        <id>https://scienceinsights.example.com/posts/ai-weather-prediction-breakthrough</id>
        <updated>2025-01-14T16:45:00Z</updated>
        <published>2025-01-14T11:30:00Z</published>
        
        <author>
            <name>Dr. James Liu</name>
            <email>james.liu@scienceinsights.example.com</email>
        </author>
        
        <category term="climate" label="Climate Science"/>
        <category term="ai" label="Artificial Intelligence"/>
        <category term="meteorology" label="Meteorology"/>
        
        <summary type="text">Revolutionary AI system achieves unprecedented accuracy in long-range weather forecasting, potentially transforming agriculture, disaster preparedness, and climate research.</summary>
        
        <content type="html"><![CDATA[
            <article>
                <h2>Transforming Weather Prediction</h2>
                
                <p>A groundbreaking artificial intelligence system developed by the Global 
                Climate Research Consortium has achieved a major milestone in meteorology: 
                accurate weather predictions up to 30 days in advance with 75% accuracy.</p>
                
                <figure>
                    <img src="/images/ai-weather-model-diagram.png" alt="AI Weather Model Architecture"/>
                    <figcaption>The neural network architecture combines satellite data, 
                    ocean temperatures, and atmospheric patterns for unprecedented forecasting accuracy.</figcaption>
                </figure>
                
                <h3>Technical Innovation</h3>
                
                <p>The system, dubbed "AtmosAI," employs a novel transformer architecture 
                specifically designed for spatiotemporal climate data. Key innovations include:</p>
                
                <ol>
                    <li><strong>Multi-Modal Data Integration:</strong> Combines satellite imagery, 
                    ocean temperature sensors, atmospheric pressure readings, and historical 
                    climate patterns</li>
                    <li><strong>Hierarchical Attention Mechanisms:</strong> Focuses on the most 
                    relevant patterns at different temporal and spatial scales</li>
                    <li><strong>Physics-Informed Learning:</strong> Incorporates fundamental 
                    atmospheric physics equations as constraints</li>
                    <li><strong>Uncertainty Quantification:</strong> Provides confidence intervals 
                    for all predictions</li>
                </ol>
                
                <h3>Real-World Impact</h3>
                
                <div class="impact-section">
                    <h4>Agriculture</h4>
                    <p>Farmers can now plan planting and harvesting schedules with unprecedented 
                    confidence, potentially increasing crop yields by 15-20%.</p>
                    
                    <h4>Disaster Preparedness</h4>
                    <p>Early warning systems for extreme weather events can now provide 
                    communities with weeks of advance notice instead of days.</p>
                    
                    <h4>Energy Management</h4>
                    <p>Power grids can optimize renewable energy integration and reduce 
                    dependency on fossil fuel backup systems.</p>
                </div>
                
                <blockquote>
                    <p>"This isn't just an incremental improvement; it's a paradigm shift 
                    in how we understand and predict Earth's atmospheric systems," said 
                    Dr. Emma Thompson, co-author of the study.</p>
                </blockquote>
            </article>
        ]]></content>
    </entry>
    
    <entry>
        <title>Exoplanet Discovery: Earth-Like World Found in Habitable Zone</title>
        <link href="https://scienceinsights.example.com/posts/earth-like-exoplanet-discovery"/>
        <id>https://scienceinsights.example.com/posts/earth-like-exoplanet-discovery</id>
        <updated>2025-01-13T09:18:00Z</updated>
        <published>2025-01-13T08:00:00Z</published>
        
        <author>
            <name>Dr. Sarah Mitchell</name>
            <email>sarah.mitchell@scienceinsights.example.com</email>
        </author>
        
        <category term="astronomy" label="Astronomy"/>
        <category term="exoplanets" label="Exoplanets"/>
        <category term="astrobiology" label="Astrobiology"/>
        
        <summary type="text">Astronomers discover Kepler-442c, an Earth-sized exoplanet in the habitable zone of its star, showing potential signs of atmospheric water vapor.</summary>
        
        <content type="html"><![CDATA[
            <article>
                <h2>A New Earth in the Cosmos</h2>
                
                <p>In a discovery that captivates the imagination and advances our search 
                for life beyond Earth, astronomers have identified Kepler-442c, an exoplanet 
                remarkably similar to our home world.</p>
                
                <h3>Planetary Characteristics</h3>
                
                <table class="planet-specs">
                    <tr><th>Property</th><th>Kepler-442c</th><th>Earth</th></tr>
                    <tr><td>Radius</td><td>1.08 Earth radii</td><td>1.0</td></tr>
                    <tr><td>Mass</td><td>1.2 Earth masses</td><td>1.0</td></tr>
                    <tr><td>Orbital Period</td><td>267 days</td><td>365 days</td></tr>
                    <tr><td>Distance from Star</td><td>0.85 AU</td><td>1.0 AU</td></tr>
                    <tr><td>Estimated Temperature</td><td>255-295 K</td><td>288 K</td></tr>
                </table>
                
                <h3>Atmospheric Analysis</h3>
                
                <p>Spectroscopic observations using the James Webb Space Telescope have 
                revealed tantalizing hints about the planet's atmosphere:</p>
                
                <ul>
                    <li><strong>Water Vapor Signatures:</strong> Consistent with liquid water on the surface</li>
                    <li><strong>Oxygen Absorption Lines:</strong> Possible indicator of photosynthetic activity</li>
                    <li><strong>Methane Traces:</strong> Could suggest biological or geological processes</li>
                    <li><strong>Cloud Formation:</strong> Evidence of dynamic weather systems</li>
                </ul>
                
                <div class="discovery-timeline">
                    <h4>Discovery Timeline</h4>
                    <ul>
                        <li><strong>2019:</strong> Initial transit detection by Kepler</li>
                        <li><strong>2023:</strong> Mass confirmation via radial velocity</li>
                        <li><strong>2024:</strong> First atmospheric observations</li>
                        <li><strong>2025:</strong> Detailed spectroscopic analysis</li>
                    </ul>
                </div>
                
                <blockquote>
                    <p>"Finding a planet so similar to Earth is like discovering a cosmic 
                    twin. The atmospheric signatures we're seeing are exactly what we'd 
                    hope to find on a potentially habitable world," remarked Dr. Michael 
                    Rodriguez, lead astronomer on the discovery team.</p>
                </blockquote>
                
                <h3>Next Steps</h3>
                
                <p>The research team plans extended observations to:</p>
                <ol>
                    <li>Confirm biosignature detections</li>
                    <li>Map seasonal atmospheric changes</li>
                    <li>Search for technosignatures</li>
                    <li>Model potential climate systems</li>
                </ol>
            </article>
        ]]></content>
    </entry>
</feed>"""

    # RSS 1.0 RDF Feed (Scientific Journal)
    samples['rss1_journal'] = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:dc="http://purl.org/dc/elements/1.1/"
         xmlns:sy="http://purl.org/rss/1.0/modules/syndication/"
         xmlns="http://purl.org/rss/1.0/">

    <channel rdf:about="https://journal.science.example.com/feed.rdf">
        <title>Advanced Materials Science Journal</title>
        <link>https://journal.science.example.com</link>
        <description>Peer-reviewed research in advanced materials science and engineering</description>
        <dc:language>en-us</dc:language>
        <dc:creator>Journal Editorial Board</dc:creator>
        <dc:publisher>Science Publications Ltd.</dc:publisher>
        <dc:date>2025-01-15T12:00:00Z</dc:date>
        <sy:updatePeriod>weekly</sy:updatePeriod>
        <sy:updateFrequency>1</sy:updateFrequency>
        
        <items>
            <rdf:Seq>
                <rdf:li resource="https://journal.science.example.com/articles/graphene-superconductor-2025"/>
                <rdf:li resource="https://journal.science.example.com/articles/metamaterial-optical-cloaking"/>
                <rdf:li resource="https://journal.science.example.com/articles/self-healing-polymers-breakthrough"/>
            </rdf:Seq>
        </items>
    </channel>

    <item rdf:about="https://journal.science.example.com/articles/graphene-superconductor-2025">
        <title>Room-Temperature Superconductivity in Twisted Graphene Multilayers</title>
        <link>https://journal.science.example.com/articles/graphene-superconductor-2025</link>
        <description>Researchers demonstrate stable room-temperature superconductivity in specially engineered twisted graphene structures, opening new possibilities for energy transmission and quantum computing.</description>
        <dc:creator>Dr. Chen Wei, Dr. Maria Gonzalez, Dr. Yuki Tanaka</dc:creator>
        <dc:subject>Superconductivity</dc:subject>
        <dc:subject>Graphene</dc:subject>
        <dc:subject>Quantum Materials</dc:subject>
        <dc:date>2025-01-15T10:00:00Z</dc:date>
        <dc:identifier>DOI:10.1000/182</dc:identifier>
    </item>

    <item rdf:about="https://journal.science.example.com/articles/metamaterial-optical-cloaking">
        <title>Broadband Optical Cloaking Using Adaptive Metamaterial Surfaces</title>
        <link>https://journal.science.example.com/articles/metamaterial-optical-cloaking</link>
        <description>Development of dynamic metamaterial surfaces capable of adaptive optical cloaking across visible light spectrum, with applications in stealth technology and optical computing.</description>
        <dc:creator>Dr. Elena Petrov, Dr. James Anderson, Dr. Liu Zhang</dc:creator>
        <dc:subject>Metamaterials</dc:subject>
        <dc:subject>Optics</dc:subject>
        <dc:subject>Nanotechnology</dc:subject>
        <dc:date>2025-01-14T14:30:00Z</dc:date>
        <dc:identifier>DOI:10.1000/183</dc:identifier>
    </item>

    <item rdf:about="https://journal.science.example.com/articles/self-healing-polymers-breakthrough">
        <title>Autonomous Self-Healing Polymers with Memory Functions</title>
        <link>https://journal.science.example.com/articles/self-healing-polymers-breakthrough</link>
        <description>Novel polymer materials that can autonomously repair damage while retaining memory of previous healing events, advancing smart materials applications.</description>
        <dc:creator>Dr. Sophie Laurent, Dr. Ahmed Hassan, Dr. Kenji Yamamoto</dc:creator>
        <dc:subject>Polymers</dc:subject>
        <dc:subject>Smart Materials</dc:subject>
        <dc:subject>Self-Healing</dc:subject>
        <dc:date>2025-01-13T09:15:00Z</dc:date>
        <dc:identifier>DOI:10.1000/184</dc:identifier>
    </item>
</rdf:RDF>"""

    # Podcast RSS Feed with Media Enclosures
    samples['podcast_feed'] = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:media="http://search.yahoo.com/mrss/">
    
    <channel>
        <title>The Future Tech Podcast</title>
        <link>https://futuretechpod.example.com</link>
        <description>Weekly discussions on emerging technologies and their impact on society</description>
        <language>en-us</language>
        <copyright>© 2025 Future Tech Media</copyright>
        <itunes:author>Dr. Alex Rivera</itunes:author>
        <itunes:summary>Weekly discussions on emerging technologies and their impact on society</itunes:summary>
        <itunes:owner>
            <itunes:n>Dr. Alex Rivera</itunes:n>
            <itunes:email>alex@futuretechpod.example.com</itunes:email>
        </itunes:owner>
        <itunes:image href="https://futuretechpod.example.com/artwork.jpg"/>
        <itunes:category text="Technology"/>
        <itunes:category text="Science"/>
        <itunes:explicit>clean</itunes:explicit>
        <pubDate>Mon, 15 Jan 2025 06:00:00 GMT</pubDate>
        <lastBuildDate>Mon, 15 Jan 2025 06:00:00 GMT</lastBuildDate>
        
        <item>
            <title>Episode 127: The Quantum Internet Revolution</title>
            <link>https://futuretechpod.example.com/episodes/127-quantum-internet</link>
            <description>Exploring the development of quantum internet infrastructure and its potential to revolutionize secure communications.</description>
            <content:encoded><![CDATA[
                <p><strong>In This Episode:</strong></p>
                <ul>
                    <li>Quantum entanglement-based communications</li>
                    <li>Current quantum network projects worldwide</li>
                    <li>Security implications and cryptography</li>
                    <li>Timeline for global quantum internet deployment</li>
                </ul>
                
                <p><strong>Guest:</strong> Dr. Maria Santos, Quantum Networks Lab</p>
                
                <h3>Show Notes</h3>
                <ul>
                    <li><a href="https://example.com/quantum-paper-2025">Recent quantum networking paper</a></li>
                    <li><a href="https://quantumnetwork.gov">National Quantum Initiative</a></li>
                    <li><a href="https://example.com/quantum-timeline">Quantum technology roadmap</a></li>
                </ul>
                
                <p><strong>Timestamps:</strong></p>
                <ul>
                    <li>00:00 - Introduction</li>
                    <li>03:15 - What is quantum internet?</li>
                    <li>12:30 - Current state of quantum networks</li>
                    <li>28:45 - Security and cryptography</li>
                    <li>41:20 - Future applications</li>
                    <li>52:10 - Listener Q&A</li>
                </ul>
            ]]></content:encoded>
            <enclosure url="https://futuretechpod.example.com/episodes/127-quantum-internet.mp3" 
                      length="67890123" type="audio/mpeg"/>
            <pubDate>Mon, 15 Jan 2025 06:00:00 GMT</pubDate>
            <guid isPermaLink="false">futuretechpod-ep127</guid>
            <itunes:author>Dr. Alex Rivera</itunes:author>
            <itunes:duration>58:34</itunes:duration>
            <itunes:explicit>clean</itunes:explicit>
            <itunes:image href="https://futuretechpod.example.com/episodes/127-artwork.jpg"/>
            <itunes:episodeType>full</itunes:episodeType>
            <itunes:episode>127</itunes:episode>
            <itunes:season>3</itunes:season>
            <media:content url="https://futuretechpod.example.com/episodes/127-quantum-internet-hd.mp4" 
                          type="video/mp4" medium="video" duration="3514"/>
        </item>
        
        <item>
            <title>Episode 126: Biocomputing and Living Processors</title>
            <link>https://futuretechpod.example.com/episodes/126-biocomputing</link>
            <description>Diving into the fascinating world of biological computing systems and DNA-based data storage.</description>
            <content:encoded><![CDATA[
                <p>This week we explore the cutting-edge field of biocomputing, where 
                biology meets information technology in revolutionary ways.</p>
                
                <h3>Topics Covered</h3>
                <ul>
                    <li>DNA data storage systems</li>
                    <li>Protein-based logic gates</li>
                    <li>Living computers using bacterial cells</li>
                    <li>Ethical considerations in biocomputing</li>
                </ul>
                
                <p><strong>Featured Research:</strong></p>
                <blockquote>
                    "We've successfully stored the entire works of Shakespeare in a test tube 
                    using synthetic DNA sequences, demonstrating the incredible density of 
                    biological data storage." - Dr. Jennifer Kim, Biocomputing Institute
                </blockquote>
                
                <p><strong>Key Statistics:</strong></p>
                <ul>
                    <li>DNA storage density: 1 exabyte per cubic millimeter</li>
                    <li>Data retention: 1000+ years without degradation</li>
                    <li>Error rates: Less than 0.001% with error correction</li>
                </ul>
            ]]></content:encoded>
            <enclosure url="https://futuretechpod.example.com/episodes/126-biocomputing.mp3" 
                      length="63245789" type="audio/mpeg"/>
            <pubDate>Mon, 08 Jan 2025 06:00:00 GMT</pubDate>
            <guid isPermaLink="false">futuretechpod-ep126</guid>
            <itunes:author>Dr. Alex Rivera</itunes:author>
            <itunes:duration>54:22</itunes:duration>
            <itunes:explicit>clean</itunes:explicit>
            <itunes:episodeType>full</itunes:episodeType>
            <itunes:episode>126</itunes:episode>
            <itunes:season>3</itunes:season>
        </item>
    </channel>
</rss>"""

    return samples

def analyze_feed_chunk(chunk) -> Dict[str, Any]:
    """Analyze the structure and properties of a feed chunk"""
    analysis = {
        'id': chunk.id,
        'type': chunk.chunk_type.value,
        'size_chars': len(chunk.content),
        'size_lines': chunk.line_count,
        'importance': chunk.importance_score,
        'tags': [tag.name for tag in chunk.semantic_tags] if hasattr(chunk, 'semantic_tags') and chunk.semantic_tags else [],
        'metadata': chunk.metadata or {},
        'dependencies': chunk.dependencies[:3] if chunk.dependencies else [],
        'preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
    }
    
    # Feed-specific analysis
    feed_type = chunk.metadata.get('feed_type') if chunk.metadata else None
    semantic_type = chunk.metadata.get('semantic_type') if chunk.metadata else None
    entry_data = chunk.metadata.get('entry_data', {}) if chunk.metadata else {}
    feed_metadata = chunk.metadata.get('feed_metadata', {}) if chunk.metadata else {}
    
    analysis['feed_analysis'] = {
        'feed_type': feed_type,
        'semantic_type': semantic_type,
        'is_feed_metadata': 'feed_metadata' in semantic_type if semantic_type else False,
        'is_entry': 'entry' in semantic_type.lower() if semantic_type else False,
        'entry_title': entry_data.get('title', ''),
        'entry_author': entry_data.get('author', ''),
        'entry_date': entry_data.get('published') or entry_data.get('pubDate', ''),
        'has_full_content': bool(entry_data.get('content')),
        'has_media': bool(entry_data.get('enclosures')),
        'content_quality': assess_content_quality(chunk.content, entry_data)
    }
    
    return analysis

def assess_content_quality(content: str, entry_data: Dict) -> str:
    """Assess the quality of feed entry content"""
    # Simple content quality scoring
    score = 0
    
    # Length factor
    if len(content) > 500:
        score += 2
    elif len(content) > 200:
        score += 1
    
    # Rich content indicators
    if entry_data.get('content') and len(entry_data['content']) > 100:
        score += 2
    
    # Multimedia content
    if entry_data.get('enclosures'):
        score += 1
    
    # Author information
    if entry_data.get('author') or entry_data.get('authors'):
        score += 1
    
    # Categories/tags
    if entry_data.get('categories'):
        score += 1
    
    if score >= 6:
        return 'High Quality'
    elif score >= 4:
        return 'Good Quality'
    elif score >= 2:
        return 'Basic Quality'
    else:
        return 'Low Quality'

def generate_feed_summary(chunks) -> Dict[str, Any]:
    """Generate summary of feed parsing results"""
    summary = {
        'total_chunks': len(chunks),
        'chunk_types': defaultdict(int),
        'feed_types': defaultdict(int),
        'content_analysis': {
            'feed_metadata_chunks': 0,
            'entry_chunks': 0,
            'total_entries': 0,
            'entries_with_full_content': 0,
            'entries_with_media': 0,
            'avg_entry_length': 0,
            'content_quality_distribution': defaultdict(int)
        },
        'temporal_analysis': {
            'date_range': {'earliest': None, 'latest': None},
            'entries_by_month': defaultdict(int),
            'publication_frequency': 'unknown'
        },
        'author_analysis': {
            'unique_authors': set(),
            'total_authored_entries': 0,
            'most_prolific_authors': defaultdict(int)
        },
        'topic_analysis': {
            'categories': defaultdict(int),
            'topics_detected': set(),
            'content_themes': defaultdict(int)
        },
        'media_analysis': {
            'total_media_items': 0,
            'audio_podcasts': 0,
            'video_content': 0,
            'images': 0,
            'other_media': 0
        }
    }
    
    total_entry_length = 0
    entry_count = 0
    
    for chunk in chunks:
        # Type distribution
        summary['chunk_types'][chunk.chunk_type.value] += 1
        
        # Analyze chunk
        analysis = analyze_feed_chunk(chunk)
        feed_analysis = analysis['feed_analysis']
        
        # Feed type distribution
        if feed_analysis['feed_type']:
            summary['feed_types'][feed_analysis['feed_type']] += 1
        
        # Content analysis
        if feed_analysis['is_feed_metadata']:
            summary['content_analysis']['feed_metadata_chunks'] += 1
        elif feed_analysis['is_entry']:
            summary['content_analysis']['entry_chunks'] += 1
            entry_count += 1
            total_entry_length += analysis['size_chars']
            
            # Content quality
            quality = feed_analysis['content_quality']
            summary['content_analysis']['content_quality_distribution'][quality] += 1
            
            # Full content and media
            if feed_analysis['has_full_content']:
                summary['content_analysis']['entries_with_full_content'] += 1
            
            if feed_analysis['has_media']:
                summary['content_analysis']['entries_with_media'] += 1
        
        # Author analysis
        entry_data = chunk.metadata.get('entry_data', {}) if chunk.metadata else {}
        author = entry_data.get('author') or (entry_data.get('authors', [{}])[0].get('name') if entry_data.get('authors') else None)
        if author:
            summary['author_analysis']['unique_authors'].add(author)
            summary['author_analysis']['total_authored_entries'] += 1
            summary['author_analysis']['most_prolific_authors'][author] += 1
        
        # Topic analysis from categories
        categories = entry_data.get('categories', [])
        for category in categories:
            if isinstance(category, dict):
                cat_text = category.get('term') or category.get('text', '')
            else:
                cat_text = str(category)
            if cat_text:
                summary['topic_analysis']['categories'][cat_text.lower()] += 1
                summary['topic_analysis']['topics_detected'].add(cat_text.lower())
        
        # Media analysis
        enclosures = entry_data.get('enclosures', [])
        for enclosure in enclosures:
            summary['media_analysis']['total_media_items'] += 1
            media_type = enclosure.get('type', '').lower()
            
            if 'audio' in media_type:
                summary['media_analysis']['audio_podcasts'] += 1
            elif 'video' in media_type:
                summary['media_analysis']['video_content'] += 1
            elif 'image' in media_type:
                summary['media_analysis']['images'] += 1
            else:
                summary['media_analysis']['other_media'] += 1
        
        # Temporal analysis
        entry_date = feed_analysis['entry_date']
        if entry_date:
            try:
                from email.utils import parsedate_to_datetime
                import dateutil.parser
                
                # Try parsing different date formats
                try:
                    dt = parsedate_to_datetime(entry_date)
                except:
                    dt = dateutil.parser.parse(entry_date)
                
                month_key = dt.strftime('%Y-%m')
                summary['temporal_analysis']['entries_by_month'][month_key] += 1
                
                # Track date range
                if not summary['temporal_analysis']['date_range']['earliest'] or dt < summary['temporal_analysis']['date_range']['earliest']:
                    summary['temporal_analysis']['date_range']['earliest'] = dt
                
                if not summary['temporal_analysis']['date_range']['latest'] or dt > summary['temporal_analysis']['date_range']['latest']:
                    summary['temporal_analysis']['date_range']['latest'] = dt
                    
            except:
                pass  # Skip invalid dates
    
    # Calculate averages
    if entry_count > 0:
        summary['content_analysis']['total_entries'] = entry_count
        summary['content_analysis']['avg_entry_length'] = total_entry_length / entry_count
    
    # Convert sets to lists for JSON serialization
    summary['author_analysis']['unique_authors'] = list(summary['author_analysis']['unique_authors'])
    summary['topic_analysis']['topics_detected'] = list(summary['topic_analysis']['topics_detected'])
    
    # Convert datetime objects to strings
    if summary['temporal_analysis']['date_range']['earliest']:
        summary['temporal_analysis']['date_range']['earliest'] = summary['temporal_analysis']['date_range']['earliest'].isoformat()
    if summary['temporal_analysis']['date_range']['latest']:
        summary['temporal_analysis']['date_range']['latest'] = summary['temporal_analysis']['date_range']['latest'].isoformat()
    
    return summary

def test_feed_parsing(content: str, sample_name: str) -> Dict[str, Any]:
    """Test RSS/Atom parsing on sample content"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    # Configure for feed parsing
    config = ChunkingConfig(
        target_chunk_size=600,  # Good size for feed entries
        min_chunk_size=100,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True
    )
    
    engine = ChunkingEngine(config)
    
    # Check if RSS/Atom is supported
    supported_languages = engine.get_supported_languages()
    rss_supported = any(lang in ['rss', 'atom', 'feed', 'syndication', 'rss_atom'] 
                       for lang in supported_languages)
    
    if not rss_supported:
        print(f"⚠️  RSS/Atom parser not available. Supported languages: {supported_languages}")
        return {
            'chunks': [],
            'summary': generate_feed_summary([]),
            'sample_info': {
                'size_chars': len(content),
                'line_count': content.count('\n') + 1,
                'error': 'RSS/Atom parser not available'
            }
        }
    
    # Determine feed type for parsing
    feed_language = 'rss_atom'  # Use the comprehensive RSS/Atom parser
    if 'atom' in sample_name:
        feed_language = 'atom'
    elif 'rss' in sample_name:
        feed_language = 'rss'
    
    # Parse feed
    start_time = time.time()
    chunks = engine.chunk_content(content, feed_language, f'{sample_name}.xml')
    parse_time = time.time() - start_time
    
    # Generate analysis
    summary = generate_feed_summary(chunks)
    
    return {
        'chunks': chunks,
        'summary': summary,
        'sample_info': {
            'size_chars': len(content),
            'line_count': content.count('\n') + 1,
            'parse_time_ms': parse_time * 1000,
            'feed_type_detected': detect_feed_type(content),
            'entries_found': summary['content_analysis']['total_entries']
        }
    }

def detect_feed_type(content: str) -> str:
    """Detect the type of feed from content"""
    content_lower = content.lower()
    
    if '<feed xmlns="http://www.w3.org/2005/Atom"' in content:
        return 'Atom 1.0'
    elif '<rss version="2.0"' in content:
        return 'RSS 2.0'
    elif '<rdf:rdf' in content_lower and 'rss/1.0' in content:
        return 'RSS 1.0 (RDF)'
    elif '<rss' in content_lower:
        return 'RSS (Unknown Version)'
    elif '<feed' in content_lower:
        return 'Atom (Unknown Version)'
    else:
        return 'Unknown Feed Format'

def demonstrate_feed_comparison():
    """Demonstrate comparison between different feed formats"""
    print(f"\n📊 FEED FORMAT COMPARISON DEMO")
    print("=" * 60)
    
    # Create minimal samples for comparison
    rss2_sample = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>RSS 2.0 Test</title>
            <description>Testing RSS 2.0 parsing</description>
            <item>
                <title>Sample Article</title>
                <description>This is a sample RSS 2.0 article</description>
                <pubDate>Mon, 15 Jan 2025 10:00:00 GMT</pubDate>
            </item>
        </channel>
    </rss>"""
    
    atom_sample = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <title>Atom 1.0 Test</title>
        <subtitle>Testing Atom 1.0 parsing</subtitle>
        <entry>
            <title>Sample Article</title>
            <summary>This is a sample Atom 1.0 article</summary>
            <published>2025-01-15T10:00:00Z</published>
        </entry>
    </feed>"""
    
    try:
        print("Comparing RSS 2.0 vs Atom 1.0 parsing...")
        
        rss_result = test_feed_parsing(rss2_sample, "rss2_comparison")
        atom_result = test_feed_parsing(atom_sample, "atom_comparison")
        
        formats = [
            ("RSS 2.0", rss_result),
            ("Atom 1.0", atom_result)
        ]
        
        print(f"\n{'Format':<12} {'Chunks':<8} {'Entries':<8} {'Parse Time':<12} {'Feed Type'}")
        print("-" * 55)
        
        for format_name, result in formats:
            if 'error' not in result['sample_info']:
                chunks = len(result['chunks'])
                entries = result['sample_info']['entries_found']
                parse_time = f"{result['sample_info']['parse_time_ms']:.1f}ms"
                feed_type = result['sample_info']['feed_type_detected']
                
                print(f"{format_name:<12} {chunks:<8} {entries:<8} {parse_time:<12} {feed_type}")
            else:
                print(f"{format_name:<12} Error: {result['sample_info']['error']}")
    
    except Exception as e:
        print(f"❌ Error in feed comparison: {e}")

def demonstrate_media_analysis():
    """Demonstrate media content analysis in podcast feeds"""
    print(f"\n🎵 PODCAST MEDIA ANALYSIS DEMO")
    print("=" * 60)
    
    try:
        samples = create_sample_feed_data()
        podcast_content = samples['podcast_feed']
        
        result = test_feed_parsing(podcast_content, "podcast_media_demo")
        
        if 'error' not in result['sample_info']:
            summary = result['summary']
            media_analysis = summary['media_analysis']
            
            print(f"✅ Podcast feed analysis complete:")
            print(f"   🎧 Total media items: {media_analysis['total_media_items']}")
            print(f"   🎵 Audio content: {media_analysis['audio_podcasts']}")
            print(f"   🎥 Video content: {media_analysis['video_content']}")
            print(f"   🖼️  Images: {media_analysis['images']}")
            print(f"   📄 Other media: {media_analysis['other_media']}")
            
            # Show episode details
            chunks = result['chunks']
            episode_chunks = [chunk for chunk in chunks 
                            if chunk.metadata and 'entry' in chunk.metadata.get('semantic_type', '').lower()]
            
            print(f"\n📻 Episode Details:")
            for chunk in episode_chunks:
                entry_data = chunk.metadata.get('entry_data', {})
                title = entry_data.get('title', 'Unknown Episode')
                duration = None
                media_urls = []
                
                # Extract duration and media URLs from enclosures
                enclosures = entry_data.get('enclosures', [])
                for enc in enclosures:
                    media_urls.append(enc.get('url', ''))
                
                print(f"   • {title}")
                if media_urls:
                    print(f"     Media: {len(media_urls)} file(s)")
                    for url in media_urls[:2]:  # Show first 2 URLs
                        media_type = url.split('.')[-1] if '.' in url else 'unknown'
                        print(f"       - {media_type.upper()}: {url.split('/')[-1]}")
        
        else:
            print(f"❌ Media analysis failed: {result['sample_info']['error']}")
    
    except Exception as e:
        print(f"❌ Error in media analysis demo: {e}")

def print_detailed_feed_analysis(samples: Dict[str, str], all_results: Dict[str, Any]):
    """Print comprehensive analysis of feed parsing results"""
    print("\n" + "="*80)
    print("🔄 COMPREHENSIVE FEED PARSING ANALYSIS")
    print("="*80)
    
    for sample_name, result in all_results.items():
        if 'error' in result['sample_info']:
            continue
            
        chunks = result['chunks']
        summary = result['summary']
        
        print(f"\n🔄 FEED: {sample_name.upper().replace('_', ' ')}")
        print("-" * 60)
        
        # Sample info
        sample_info = result['sample_info']
        print(f"📄 Feed size: {sample_info['size_chars']} characters, {sample_info['line_count']} lines")
        print(f"🧩 Chunks created: {len(chunks)}")
        print(f"⏱️ Parse time: {sample_info['parse_time_ms']:.1f}ms")
        print(f"📰 Feed type: {sample_info['feed_type_detected']}")
        print(f"📝 Entries found: {sample_info['entries_found']}")
        
        # Content analysis
        content_analysis = summary['content_analysis']
        print(f"\n📊 Content Analysis:")
        print(f"   Feed metadata chunks: {content_analysis['feed_metadata_chunks']}")
        print(f"   Entry chunks: {content_analysis['entry_chunks']}")
        print(f"   Entries with full content: {content_analysis['entries_with_full_content']}")
        print(f"   Entries with media: {content_analysis['entries_with_media']}")
        if content_analysis['avg_entry_length'] > 0:
            print(f"   Average entry length: {content_analysis['avg_entry_length']:.0f} characters")
        
        # Content quality distribution
        quality_dist = content_analysis['content_quality_distribution']
        if quality_dist:
            print(f"   Content quality:")
            for quality, count in quality_dist.items():
                print(f"     {quality}: {count} entries")
        
        # Author analysis
        author_analysis = summary['author_analysis']
        if author_analysis['unique_authors']:
            print(f"\n👥 Author Analysis:")
            print(f"   Unique authors: {len(author_analysis['unique_authors'])}")
            print(f"   Total authored entries: {author_analysis['total_authored_entries']}")
            
            # Most prolific authors
            prolific = sorted(author_analysis['most_prolific_authors'].items(), 
                            key=lambda x: x[1], reverse=True)
            if prolific:
                print(f"   Most prolific authors:")
                for author, count in prolific[:3]:
                    print(f"     {author}: {count} entries")
        
        # Topic analysis
        topic_analysis = summary['topic_analysis']
        if topic_analysis['categories']:
            print(f"\n🏷️ Topic Analysis:")
            print(f"   Total categories: {len(topic_analysis['topics_detected'])}")
            
            # Most common topics
            common_topics = sorted(topic_analysis['categories'].items(), 
                                 key=lambda x: x[1], reverse=True)
            if common_topics:
                print(f"   Most common topics:")
                for topic, count in common_topics[:5]:
                    print(f"     {topic}: {count} mentions")
        
        # Media analysis
        media_analysis = summary['media_analysis']
        if media_analysis['total_media_items'] > 0:
            print(f"\n🎵 Media Analysis:")
            print(f"   Total media items: {media_analysis['total_media_items']}")
            print(f"   Audio/podcasts: {media_analysis['audio_podcasts']}")
            print(f"   Video content: {media_analysis['video_content']}")
            print(f"   Images: {media_analysis['images']}")
            print(f"   Other media: {media_analysis['other_media']}")
        
        # Temporal analysis
        temporal_analysis = summary['temporal_analysis']
        if temporal_analysis['date_range']['earliest']:
            print(f"\n📅 Temporal Analysis:")
            print(f"   Date range: {temporal_analysis['date_range']['earliest'][:10]} to {temporal_analysis['date_range']['latest'][:10]}")
            
            entries_by_month = temporal_analysis['entries_by_month']
            if entries_by_month:
                print(f"   Entries by month:")
                for month, count in sorted(entries_by_month.items())[-3:]:  # Show last 3 months
                    print(f"     {month}: {count} entries")

def main():
    """Main demo function"""
    print("🔄 RSS/ATOM FEED PARSING DEMO")
    print("="*60)
    
    try:
        # Import test
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        print("✅ Chunking system imported successfully")
        
        # Test engine initialization and parser availability
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        supported_languages = engine.get_supported_languages()
        print(f"✅ Engine initialized with support for: {', '.join(supported_languages)}")
        
        rss_supported = any(lang in ['rss', 'atom', 'feed', 'syndication', 'rss_atom'] 
                           for lang in supported_languages)
        
        if not rss_supported:
            print("⚠️  RSS/Atom parser not available. This may be due to missing tree-sitter-xml.")
            print("   Install with: pip install tree-sitter-xml")
            print("   Demo will show what the analysis would look like.")
        
        # Create sample feed data
        print(f"\n📝 Creating sample feed data...")
        samples = create_sample_feed_data()
        print(f"✅ Created {len(samples)} feed samples:")
        for name, content in samples.items():
            lines = content.count('\n') + 1
            chars = len(content)
            feed_type = detect_feed_type(content)
            print(f"   • {name}: {chars:,} chars, {lines} lines ({feed_type})")
        
        # Test each sample
        all_results = {}
        
        print(f"\n🚀 Testing feed parsing...")
        for sample_name, sample_content in samples.items():
            print(f"   🧪 Processing {sample_name}...")
            
            try:
                result = test_feed_parsing(sample_content, sample_name)
                all_results[sample_name] = result
                
                if 'error' in result['sample_info']:
                    print(f"      ⚠️  {result['sample_info']['error']}")
                else:
                    chunks_count = len(result['chunks'])
                    parse_time = result['sample_info']['parse_time_ms']
                    entries = result['sample_info']['entries_found']
                    feed_type = result['sample_info']['feed_type_detected']
                    
                    print(f"      ✅ {chunks_count} chunks, {entries} entries, {parse_time:.1f}ms ({feed_type})")
                    
            except Exception as e:
                print(f"      ❌ Error processing {sample_name}: {e}")
        
        # Print comprehensive analysis
        print_detailed_feed_analysis(samples, all_results)
        
        # Demonstratez special features
        demonstrate_feed_comparison()
        demonstrate_media_analysis()
        
        print(f"\n🎉 RSS/Atom feed parsing demo completed successfully!")
        print(f"💡 The RSS/Atom parser provides sophisticated analysis of syndication feeds")
        print(f"   with semantic understanding, media detection, and temporal analysis.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-xml are installed")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()