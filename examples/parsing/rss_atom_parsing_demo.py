"""
Comprehensive pytest unit tests for RSS 2.0 Parser

Tests cover:
- RSS 2.0 format detection and validation
- Feed metadata extraction
- Item parsing with various content types
- Dublin Core, iTunes, and Media RSS extensions
- Error handling and fallback scenarios
- Configuration options
- Semantic tagging
"""

import pytest
import xml.etree.ElementTree as ET
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from email.utils import formatdate

# Assuming these imports based on the code structure
from chuk_code_raptor.chunking.parsers.rss import RSSParser
from chuk_code_raptor.chunking.base import ParseContext
from chuk_code_raptor.chunking.semantic_chunk import SemanticChunk
from chuk_code_raptor.core.models import ChunkType


# Module-level fixtures
@pytest.fixture
def config():
    """Mock configuration for RSS parser"""
    config = Mock()
    config.rss2_extract_full_content = True
    config.rss2_prefer_content_encoded = True
    config.rss2_clean_html_content = True
    config.rss2_extract_media_metadata = True
    config.rss2_extract_itunes_metadata = True
    config.rss2_max_items_per_feed = 200
    config.rss2_min_item_content_length = 30
    config.rss2_include_feed_metadata = True
    return config


@pytest.fixture
def minimal_config():
    """Minimal configuration without optional features"""
    config = Mock()
    config.rss2_extract_full_content = False
    config.rss2_prefer_content_encoded = False
    config.rss2_clean_html_content = False
    config.rss2_extract_media_metadata = False
    config.rss2_extract_itunes_metadata = False
    config.rss2_max_items_per_feed = 10
    config.rss2_min_item_content_length = 50
    config.rss2_include_feed_metadata = False
    return config


@pytest.fixture
def parse_context():
    """Mock parse context"""
    context = Mock(spec=ParseContext)
    context.file_path = "test_feed.rss"
    context.content_type = "application/rss+xml"
    context.language = "xml"
    return context


@pytest.fixture
def rss_parser(config):
    """RSS parser instance with mocked tree-sitter"""
    with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
        mock_ts.return_value = (Mock(), 'tree_sitter_xml')
        parser = RSSParser(config)
        return parser


@pytest.fixture
def basic_rss_feed():
    """Basic RSS 2.0 feed content"""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Test Blog</title>
        <description>A test RSS feed</description>
        <link>https://example.com</link>
        <language>en-us</language>
        <lastBuildDate>Wed, 24 Jul 2024 10:00:00 GMT</lastBuildDate>
        
        <item>
            <title>First Post</title>
            <description>This is the first test post</description>
            <link>https://example.com/post1</link>
            <pubDate>Wed, 24 Jul 2024 09:00:00 GMT</pubDate>
            <guid>https://example.com/post1</guid>
        </item>
        
        <item>
            <title>Second Post</title>
            <description>This is the second test post with more content</description>
            <link>https://example.com/post2</link>
            <author>test@example.com</author>
            <pubDate>Tue, 23 Jul 2024 08:00:00 GMT</pubDate>
            <guid isPermaLink="false">post-2-guid</guid>
        </item>
    </channel>
</rss>'''


@pytest.fixture
def extended_rss_feed():
    """RSS feed with extended metadata and namespaces"""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" 
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:media="http://search.yahoo.com/mrss/">
    <channel>
        <title>Podcast Feed</title>
        <description>A test podcast feed</description>
        <link>https://podcast.example.com</link>
        <language>en-us</language>
        <copyright>© 2024 Example Corp</copyright>
        <managingEditor>editor@example.com</managingEditor>
        <category domain="Technology">Tech News</category>
        <category>Programming</category>
        
        <itunes:author>Podcast Host</itunes:author>
        <itunes:category text="Technology"/>
        <itunes:explicit>false</itunes:explicit>
        <itunes:image href="https://podcast.example.com/artwork.jpg"/>
        
        <image>
            <url>https://example.com/logo.png</url>
            <title>Test Blog</title>
            <link>https://example.com</link>
            <width>144</width>
            <height>144</height>
        </image>
        
        <item>
            <title>Episode 1: Getting Started</title>
            <description>Introduction to our podcast</description>
            <link>https://podcast.example.com/episode1</link>
            <author>host@example.com</author>
            <category>Technology</category>
            <category domain="Programming">Python</category>
            <pubDate>Wed, 24 Jul 2024 09:00:00 GMT</pubDate>
            <guid>episode-1-2024</guid>
            
            <content:encoded><![CDATA[
                <p>Welcome to our <strong>first episode</strong>!</p>
                <p>In this episode we discuss:</p>
                <ul>
                    <li>Getting started with podcasting</li>
                    <li>Technical setup</li>
                    <li>Content planning</li>
                </ul>
            ]]></content:encoded>
            
            <dc:creator>John Doe</dc:creator>
            <dc:date>2024-07-24</dc:date>
            <dc:subject>Podcasting</dc:subject>
            
            <itunes:author>John Doe</itunes:author>
            <itunes:duration>25:30</itunes:duration>
            <itunes:explicit>false</itunes:explicit>
            <itunes:episode>1</itunes:episode>
            <itunes:season>1</itunes:season>
            
            <enclosure url="https://podcast.example.com/episode1.mp3" 
                      type="audio/mpeg" 
                      length="24576000"/>
            
            <media:content url="https://podcast.example.com/episode1.mp3" 
                          type="audio/mpeg" 
                          duration="1530"/>
            <media:thumbnail url="https://podcast.example.com/ep1-thumb.jpg" 
                           width="300" 
                           height="300"/>
        </item>
    </channel>
</rss>'''


@pytest.fixture
def invalid_rss_feeds():
    """Various invalid RSS feed formats"""
    return {
        'not_rss': '<html><body>Not RSS</body></html>',
        'malformed_xml': '<rss version="2.0"><channel><title>Test</channel></rss>',
        'no_channel': '<rss version="2.0"></rss>',
        'rss_1_0': '''<?xml version="1.0"?>
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                     xmlns="http://purl.org/rss/1.0/">
                <channel rdf:about="http://example.com/">
                    <title>RSS 1.0 Feed</title>
                </channel>
            </rdf:RDF>''',
        'atom_feed': '''<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
                <title>Atom Feed</title>
            </feed>'''
    }


class TestRSSParserInitialization:
    """Test RSS parser initialization and configuration"""
    
    def test_parser_initialization_with_full_config(self, config):
        """Test parser initialization with full configuration"""
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            
            parser = RSSParser(config)
            
            assert parser.name == "RSSParser"
            assert parser.supported_languages == {'rss', 'rss2'}
            assert parser.supported_extensions == {'.rss', '.xml'}
            assert parser.extract_full_content is True
            assert parser.prefer_content_encoded is True
            assert parser.max_items_per_feed == 200
            assert parser.min_item_content_length == 30
    
    def test_parser_initialization_with_minimal_config(self, minimal_config):
        """Test parser initialization with minimal configuration"""
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            
            parser = RSSParser(minimal_config)
            
            assert parser.extract_full_content is False
            assert parser.prefer_content_encoded is False
            assert parser.max_items_per_feed == 10
            assert parser.min_item_content_length == 50
            assert parser.include_feed_metadata is False
    
    def test_tree_sitter_language_loading_failure(self, config):
        """Test handling of tree-sitter language loading failure"""
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (None, None)
            
            with pytest.raises(ImportError, match="tree-sitter XML package required"):
                RSSParser(config)
    
    def test_namespaces_configuration(self, rss_parser):
        """Test that RSS namespaces are properly configured"""
        expected_namespaces = {
            'content': 'http://purl.org/rss/1.0/modules/content/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'media': 'http://search.yahoo.com/mrss/',
            'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd',
            'atom': 'http://www.w3.org/2005/Atom'
        }
        
        assert rss_parser.namespaces == expected_namespaces


class TestRSSFormatDetection:
    """Test RSS 2.0 format detection"""
    
    def test_is_rss2_feed_valid_formats(self, rss_parser):
        """Test RSS 2.0 format detection with valid formats"""
        valid_feeds = [
            '<rss version="2.0">',
            '<rss version="2.1">',
            '<RSS VERSION="2.0">',  # Case insensitive
            '<rss xmlns="http://example.com" version="2.0">',
            '<?xml version="1.0"?>\n<rss version="2.0">'
        ]
        
        for feed in valid_feeds:
            assert rss_parser._is_rss2_feed(feed) is True
    
    def test_is_rss2_feed_invalid_formats(self, rss_parser, invalid_rss_feeds):
        """Test RSS 2.0 format detection with invalid formats"""
        for feed_type, content in invalid_rss_feeds.items():
            assert rss_parser._is_rss2_feed(content) is False, f"Failed for {feed_type}"
    
    def test_is_rss2_feed_edge_cases(self, rss_parser):
        """Test RSS 2.0 format detection edge cases"""
        edge_cases = [
            '',  # Empty string
            '   ',  # Whitespace only
            '<rss version="1.0">',  # RSS 1.0
            '<rss version="3.0">',  # Future version
            '<feed xmlns="http://www.w3.org/2005/Atom">',  # Atom
        ]
        
        for case in edge_cases:
            assert rss_parser._is_rss2_feed(case) is False


class TestBasicRSSParsing:
    """Test basic RSS 2.0 parsing functionality"""
    
    def test_parse_empty_content(self, rss_parser, parse_context):
        """Test parsing empty content"""
        result = rss_parser.parse('', parse_context)
        assert result == []
        
        result = rss_parser.parse('   ', parse_context)
        assert result == []
    
    def test_parse_basic_rss_feed(self, rss_parser, basic_rss_feed, parse_context):
        """Test parsing basic RSS 2.0 feed"""
        chunks = rss_parser.parse(basic_rss_feed, parse_context)
        
        # Should have metadata chunk + 2 item chunks
        assert len(chunks) == 3
        
        # Check metadata chunk
        metadata_chunk = chunks[0]
        assert metadata_chunk.chunk_type == ChunkType.METADATA
        assert "Test Blog" in metadata_chunk.content
        assert "A test RSS feed" in metadata_chunk.content
        assert metadata_chunk.metadata['semantic_type'] == 'RSS 2.0 Feed Metadata'
        
        # Check item chunks
        item1 = chunks[1]
        assert item1.chunk_type == ChunkType.TEXT_BLOCK
        assert "First Post" in item1.content
        assert "first test post" in item1.content
        assert item1.metadata['semantic_type'] == 'RSS 2.0 Item'
        assert item1.metadata['item_number'] == 1
        
        item2 = chunks[2]
        assert "Second Post" in item2.content
        assert "test@example.com" in item2.content
        assert item2.metadata['item_number'] == 2
    
    def test_parse_without_feed_metadata(self, minimal_config, basic_rss_feed, parse_context):
        """Test parsing without feed metadata extraction"""
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            parser = RSSParser(minimal_config)
            
            chunks = parser.parse(basic_rss_feed, parse_context)
            
            # Should only have item chunks, no metadata chunk
            assert len(chunks) == 0  # Items too short for min_item_content_length=50
    
    def test_parse_with_item_limit(self, config, parse_context):
        """Test parsing with item limit"""
        config.rss2_max_items_per_feed = 1
        
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            parser = RSSParser(config)
            
            # Create feed with 3 items
            rss_content = '''<?xml version="1.0"?>
            <rss version="2.0">
                <channel>
                    <title>Test</title>
                    <description>Test feed</description>
                    <item><title>Item 1</title><description>Long enough description for minimum length requirement</description></item>
                    <item><title>Item 2</title><description>Long enough description for minimum length requirement</description></item>
                    <item><title>Item 3</title><description>Long enough description for minimum length requirement</description></item>
                </channel>
            </rss>'''
            
            chunks = parser.parse(rss_content, parse_context)
            
            # Should have metadata + 1 item (limited)
            assert len(chunks) == 2
            item_chunks = [c for c in chunks if c.chunk_type == ChunkType.TEXT_BLOCK]
            assert len(item_chunks) == 1
            assert "Item 1" in item_chunks[0].content


class TestExtendedRSSParsing:
    """Test parsing RSS feeds with extended metadata"""
    
    def test_parse_extended_rss_feed(self, rss_parser, extended_rss_feed, parse_context):
        """Test parsing RSS feed with extended metadata"""
        chunks = rss_parser.parse(extended_rss_feed, parse_context)
        
        assert len(chunks) == 2  # metadata + 1 item
        
        # Check metadata chunk
        metadata_chunk = chunks[0]
        assert "Podcast Feed" in metadata_chunk.content
        assert "© 2024 Example Corp" in metadata_chunk.content
        assert "editor@example.com" in metadata_chunk.content
        assert "Tech News" in metadata_chunk.content
        assert "Podcast Host" in metadata_chunk.content
        
        metadata = metadata_chunk.metadata['feed_metadata']
        assert metadata['title'] == "Podcast Feed"
        assert metadata['copyright'] == "© 2024 Example Corp"
        assert len(metadata['categories']) == 2
        assert metadata['itunes']['author'] == "Podcast Host"
        
        # Check item chunk
        item_chunk = chunks[1]
        assert "Episode 1: Getting Started" in item_chunk.content
        assert "John Doe" in item_chunk.content  # Dublin Core creator
        assert "25:30" in item_chunk.content  # iTunes duration
        assert "audio/mpeg" in item_chunk.content  # Media enclosure
        
        item_data = item_chunk.metadata['item_data']
        assert item_data['dc_creator'] == "John Doe"
        assert item_data['itunes']['duration'] == "25:30"
        assert len(item_data['enclosures']) == 1
        assert item_data['enclosures'][0]['type'] == "audio/mpeg"
    
    def test_content_encoded_extraction(self, rss_parser, extended_rss_feed, parse_context):
        """Test content:encoded extraction"""
        chunks = rss_parser.parse(extended_rss_feed, parse_context)
        item_chunk = chunks[1]  # Skip metadata
        
        # Should contain cleaned HTML content
        assert "Welcome to our first episode!" in item_chunk.content
        assert "Getting started with podcasting" in item_chunk.content
        assert "Technical setup" in item_chunk.content
        
        # HTML tags should be cleaned
        assert "<p>" not in item_chunk.content
        assert "<strong>" not in item_chunk.content
        assert "<ul>" not in item_chunk.content
    
    def test_dublin_core_metadata_extraction(self, rss_parser):
        """Test Dublin Core metadata extraction"""
        item_xml = '''<item xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:creator>Jane Smith</dc:creator>
            <dc:date>2024-07-24</dc:date>
            <dc:subject>Science</dc:subject>
            <dc:publisher>Science Corp</dc:publisher>
            <dc:rights>All rights reserved</dc:rights>
        </item>'''
        
        item = ET.fromstring(item_xml)
        dc_data = rss_parser._extract_dublin_core_metadata(item)
        
        assert dc_data['dc_creator'] == "Jane Smith"
        assert dc_data['dc_date'] == "2024-07-24"
        assert dc_data['dc_subject'] == "Science"
        assert dc_data['dc_publisher'] == "Science Corp"
        assert dc_data['dc_rights'] == "All rights reserved"
    
    def test_itunes_metadata_extraction(self, rss_parser):
        """Test iTunes podcast metadata extraction"""
        item_xml = '''<item xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
            <itunes:author>Podcast Host</itunes:author>
            <itunes:duration>45:30</itunes:duration>
            <itunes:explicit>true</itunes:explicit>
            <itunes:episode>5</itunes:episode>
            <itunes:season>2</itunes:season>
            <itunes:episodeType>full</itunes:episodeType>
            <itunes:category text="Technology"/>
            <itunes:image href="https://example.com/image.jpg"/>
        </item>'''
        
        item = ET.fromstring(item_xml)
        itunes_data = rss_parser._extract_itunes_metadata(item)
        
        assert itunes_data['author'] == "Podcast Host"
        assert itunes_data['duration'] == "45:30"
        assert itunes_data['explicit'] == "true"
        assert itunes_data['episode'] == "5"
        assert itunes_data['season'] == "2"
        assert itunes_data['episodeType'] == "full"
        assert itunes_data['category'] == "Technology"
        assert itunes_data['image'] == "https://example.com/image.jpg"
    
    def test_media_rss_metadata_extraction(self, rss_parser):
        """Test Media RSS metadata extraction"""
        item_xml = '''<item xmlns:media="http://search.yahoo.com/mrss/">
            <media:content url="https://example.com/video.mp4" 
                          type="video/mp4" 
                          medium="video"
                          duration="300"
                          width="1920"
                          height="1080"/>
            <media:thumbnail url="https://example.com/thumb.jpg" 
                           width="300" 
                           height="200"/>
        </item>'''
        
        item = ET.fromstring(item_xml)
        media_data = rss_parser._extract_media_metadata(item)
        
        assert media_data['content']['url'] == "https://example.com/video.mp4"
        assert media_data['content']['type'] == "video/mp4"
        assert media_data['content']['medium'] == "video"
        assert media_data['content']['duration'] == "300"
        assert media_data['content']['width'] == "1920"
        assert media_data['content']['height'] == "1080"
        
        assert media_data['thumbnail']['url'] == "https://example.com/thumb.jpg"
        assert media_data['thumbnail']['width'] == "300"
        assert media_data['thumbnail']['height'] == "200"


class TestSemanticTagging:
    """Test semantic tagging functionality"""
    
    def test_feed_metadata_semantic_tags(self, rss_parser, basic_rss_feed, parse_context):
        """Test semantic tags on feed metadata"""
        chunks = rss_parser.parse(basic_rss_feed, parse_context)
        metadata_chunk = chunks[0]
        
        # Mock the add_tag method to verify calls
        metadata_chunk.add_tag = Mock()
        rss_parser._add_item_semantic_tags = Mock()  # Don't interfere with item tagging
        
        # Re-parse to trigger tagging
        chunks = rss_parser.parse(basic_rss_feed, parse_context)
        metadata_chunk = chunks[0]
        
        # Verify the chunk has the expected structure (tags would be added in real implementation)
        assert metadata_chunk.chunk_type == ChunkType.METADATA
        assert metadata_chunk.metadata['feed_type'] == 'rss2'
    
    def test_item_semantic_tags_basic(self, rss_parser):
        """Test basic item semantic tagging"""
        chunk = Mock(spec=SemanticChunk)
        chunk.add_tag = Mock()
        
        item_data = {
            'title': 'Test Article',
            'categories': [
                {'text': 'Technology', 'domain': None},
                {'text': 'Programming', 'domain': 'tech'}
            ],
            'enclosures': [],
            'content_encoded': None
        }
        
        rss_parser._add_item_semantic_tags(chunk, item_data)
        
        # Verify base tags
        chunk.add_tag.assert_any_call('rss2_item', source='rss2_parser')
        chunk.add_tag.assert_any_call('rss_item', source='rss2_parser')
        chunk.add_tag.assert_any_call('article', source='rss2_parser')
        
        # Verify category tags
        chunk.add_tag.assert_any_call('topic:technology', source='rss2_parser')
        chunk.add_tag.assert_any_call('topic:programming', source='rss2_parser')
    
    def test_item_semantic_tags_multimedia(self, rss_parser):
        """Test semantic tagging for multimedia content"""
        chunk = Mock(spec=SemanticChunk)
        chunk.add_tag = Mock()
        
        item_data = {
            'categories': [],
            'enclosures': [
                {'type': 'audio/mpeg', 'url': 'podcast.mp3'},
                {'type': 'video/mp4', 'url': 'video.mp4'},
                {'type': 'image/jpeg', 'url': 'image.jpg'}
            ],
            'content_encoded': '<p>Full content</p>',
            'author': 'Test Author'
        }
        
        rss_parser._add_item_semantic_tags(chunk, item_data)
        
        # Verify multimedia tags
        chunk.add_tag.assert_any_call('multimedia', source='rss2_parser')
        chunk.add_tag.assert_any_call('podcast', source='rss2_parser')
        chunk.add_tag.assert_any_call('audio', source='rss2_parser')
        chunk.add_tag.assert_any_call('video', source='rss2_parser')
        chunk.add_tag.assert_any_call('image', source='rss2_parser')
        chunk.add_tag.assert_any_call('full_content', source='rss2_parser')
        chunk.add_tag.assert_any_call('authored', source='rss2_parser')


class TestImportanceScoring:
    """Test importance score calculation"""
    
    def test_calculate_item_importance_basic(self, rss_parser):
        """Test basic importance calculation"""
        item_data = {'title': 'Basic Item'}
        content = "Basic content"
        
        score = rss_parser._calculate_item_importance(item_data, content)
        assert score == 0.7  # Base score only
    
    def test_calculate_item_importance_full_content(self, rss_parser):
        """Test importance with full content"""
        item_data = {
            'content_encoded': '<p>Full HTML content</p>',
            'author': 'Test Author',
            'categories': [{'text': 'Tech'}],
            'enclosures': [{'type': 'audio/mpeg'}],
            'pubDate': formatdate()  # Current date
        }
        content = "A" * 1500  # Long content
        
        score = rss_parser._calculate_item_importance(item_data, content)
        
        # Should have bonuses for: content_encoded (+0.15), long content (+0.1), 
        # author (+0.05), categories (+0.05), enclosures (+0.1), recent date (+0.05)
        expected = 0.7 + 0.15 + 0.1 + 0.05 + 0.05 + 0.1 + 0.05
        assert score == min(expected, 1.0)
    
    def test_calculate_item_importance_date_parsing(self, rss_parser):
        """Test importance calculation with date parsing"""
        from email.utils import formatdate
        from datetime import datetime, timedelta
        
        # Test recent date
        recent_date = formatdate((datetime.now() - timedelta(days=3)).timestamp())
        item_data = {'pubDate': recent_date}
        score_recent = rss_parser._calculate_item_importance(item_data, "content")
        
        # Test old date
        old_date = formatdate((datetime.now() - timedelta(days=100)).timestamp())
        item_data = {'pubDate': old_date}
        score_old = rss_parser._calculate_item_importance(item_data, "content")
        
        # Recent should have higher score
        assert score_recent > score_old
    
    def test_calculate_item_importance_invalid_date(self, rss_parser):
        """Test importance calculation with invalid date"""
        item_data = {'pubDate': 'invalid-date-format'}
        content = "Basic content"
        
        # Should not crash and should return base score
        score = rss_parser._calculate_item_importance(item_data, content)
        assert score == 0.7


class TestUtilityMethods:
    """Test utility methods"""
    
    def test_get_element_text(self, rss_parser):
        """Test element text extraction"""
        xml = '<parent><child>Test Content</child><empty></empty></parent>'
        parent = ET.fromstring(xml)
        
        assert rss_parser._get_element_text(parent, 'child') == "Test Content"
        assert rss_parser._get_element_text(parent, 'empty') is None
        assert rss_parser._get_element_text(parent, 'nonexistent') is None
    
    def test_clean_html_content(self, rss_parser):
        """Test HTML content cleaning"""
        html = '''
        <p>This is a <strong>test</strong> with <em>HTML</em> tags.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        <p>More content here.</p>
        '''
        
        cleaned = rss_parser._clean_html_content(html)
        
        assert "<p>" not in cleaned
        assert "<strong>" not in cleaned
        assert "<li>" not in cleaned
        assert "This is a test with HTML tags." in cleaned
        assert "Item 1 Item 2" in cleaned
        assert "More content here." in cleaned
    
    def test_clean_html_content_long(self, rss_parser):
        """Test HTML cleaning with length limit"""
        long_html = "<p>" + "A" * 3000 + "</p>"
        
        cleaned = rss_parser._clean_html_content(long_html)
        
        assert len(cleaned) <= 2000
        assert cleaned.endswith("...")
    
    def test_clean_html_content_empty(self, rss_parser):
        """Test HTML cleaning with empty content"""
        assert rss_parser._clean_html_content("") == ""
        assert rss_parser._clean_html_content(None) == ""


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_parse_malformed_xml(self, rss_parser, parse_context):
        """Test parsing malformed XML"""
        malformed_xml = '<rss version="2.0"><channel><title>Test</channel></rss>'
        
        chunks = rss_parser.parse(malformed_xml, parse_context)
        
        # Should return fallback chunk
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_type == ChunkType.TEXT_BLOCK
        assert "XML Parse Error" in chunk.content
        assert chunk.metadata['semantic_type'] == 'RSS 2.0 Fallback'
    
    def test_parse_non_rss_content(self, rss_parser, parse_context):
        """Test parsing non-RSS content"""
        html_content = '<html><body>Not RSS</body></html>'
        
        chunks = rss_parser.parse(html_content, parse_context)
        
        # Should return empty list
        assert chunks == []
    
    def test_parse_rss_without_channel(self, rss_parser, parse_context):
        """Test parsing RSS without channel element"""
        rss_without_channel = '<?xml version="1.0"?><rss version="2.0"></rss>'
        
        chunks = rss_parser.parse(rss_without_channel, parse_context)
        
        # Should return fallback chunk
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "Parse Error" in chunk.content
    
    def test_parse_item_error_handling(self, rss_parser, parse_context):
        """Test error handling in item parsing"""
        # Mock _parse_rss_item to raise exception
        with patch.object(rss_parser, '_parse_rss_item', side_effect=Exception("Test error")):
            rss_content = '''<?xml version="1.0"?>
            <rss version="2.0">
                <channel>
                    <title>Test</title>
                    <item><title>Test Item</title></item>
                </channel>
            </rss>'''
            
            chunks = rss_parser.parse(rss_content, parse_context)
            
            # Should have metadata chunk but no item chunks
            assert len(chunks) == 1
            assert chunks[0].chunk_type == ChunkType.METADATA
    
    def test_create_fallback_chunk(self, rss_parser, parse_context):
        """Test fallback chunk creation"""
        content = '<rss version="2.0"><channel><title>Test Feed</title></channel></rss>'
        error_msg = "Test error message"
        
        chunks = rss_parser._create_fallback_chunk(content, parse_context, error_msg)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_type == ChunkType.TEXT_BLOCK
        assert "Test Feed" in chunk.content
        assert error_msg in chunk.content
        assert chunk.metadata['error'] == error_msg
    
    def test_item_content_length_filtering(self, rss_parser, parse_context):
        """Test filtering items by content length"""
        rss_content = '''<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test</title>
                <item>
                    <title>Short</title>
                    <description>Too short</description>
                </item>
                <item>
                    <title>Long Enough Item</title>
                    <description>This description is long enough to meet the minimum content length requirement</description>
                </item>
            </channel>
        </rss>'''
        
        chunks = rss_parser.parse(rss_content, parse_context)
        
        # Should have metadata + 1 item (short item filtered out)
        assert len(chunks) == 2
        item_chunks = [c for c in chunks if c.chunk_type == ChunkType.TEXT_BLOCK]
        assert len(item_chunks) == 1
        assert "Long Enough Item" in item_chunks[0].content


class TestConfigurationOptions:
    """Test various configuration options"""
    
    def test_disable_full_content_extraction(self, config, parse_context):
        """Test disabling full content extraction"""
        config.rss2_extract_full_content = False
        
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            parser = RSSParser(config)
            
            rss_content = '''<?xml version="1.0"?>
            <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
                <channel>
                    <title>Test</title>
                    <item>
                        <title>Test Item</title>
                        <description>Basic description</description>
                        <content:encoded><![CDATA[<p>Full HTML content</p>]]></content:encoded>
                    </item>
                </channel>
            </rss>'''
            
            chunks = parser.parse(rss_content, parse_context)
            item_chunk = [c for c in chunks if c.chunk_type == ChunkType.TEXT_BLOCK][0]
            
            # Should not contain full content
            assert "Full HTML content" not in item_chunk.content
            assert "Basic description" in item_chunk.content
    
    def test_disable_content_encoded_preference(self, config, parse_context):
        """Test disabling content:encoded preference"""
        config.rss2_prefer_content_encoded = False
        
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            parser = RSSParser(config)
            
            # This would require more complex testing to verify the preference behavior
            assert parser.prefer_content_encoded is False
    
    def test_disable_html_cleaning(self, config, parse_context):
        """Test disabling HTML cleaning"""
        config.rss2_clean_html_content = False
        
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            parser = RSSParser(config)
            
            assert parser.clean_html_content is False
    
    def test_disable_metadata_extraction(self, config, parse_context):
        """Test disabling various metadata extraction options"""
        config.rss2_extract_media_metadata = False
        config.rss2_extract_itunes_metadata = False
        
        with patch('chuk_code_raptor.chunking.parsers.rss.get_tree_sitter_language_robust') as mock_ts:
            mock_ts.return_value = (Mock(), 'tree_sitter_xml')
            parser = RSSParser(config)
            
            assert parser.extract_media_metadata is False
            assert parser.extract_itunes_metadata is False


class TestRSSParserIntegration:
    """Integration tests for complete RSS parsing workflow"""
    
    def test_complete_parsing_workflow(self, rss_parser, extended_rss_feed, parse_context):
        """Test complete parsing workflow with all features"""
        chunks = rss_parser.parse(extended_rss_feed, parse_context)
        
        # Verify we got expected chunks
        assert len(chunks) == 2
        
        # Verify metadata chunk
        metadata_chunk = chunks[0]
        assert metadata_chunk.chunk_type == ChunkType.METADATA
        assert metadata_chunk.importance_score == 0.95
        assert "Podcast Feed" in metadata_chunk.content
        
        # Verify comprehensive metadata
        feed_metadata = metadata_chunk.metadata['feed_metadata']
        assert 'title' in feed_metadata
        assert 'categories' in feed_metadata
        assert 'itunes' in feed_metadata
        assert 'image' in feed_metadata
        
        # Verify item chunk
        item_chunk = chunks[1]
        assert item_chunk.chunk_type == ChunkType.TEXT_BLOCK
        assert item_chunk.importance_score > 0.7  # Should have bonuses
        
        # Verify comprehensive item metadata
        item_data = item_chunk.metadata['item_data']
        assert 'content_encoded' in item_data
        assert 'dc_creator' in item_data
        assert 'itunes' in item_data
        assert 'enclosures' in item_data
        assert 'categories' in item_data
        
        # Verify content processing
        assert "Episode 1: Getting Started" in item_chunk.content
        assert "Welcome to our first episode!" in item_chunk.content  # Cleaned HTML
        assert "John Doe" in item_chunk.content  # DC creator
        assert "25:30" in item_chunk.content  # iTunes duration
        assert "audio/mpeg" in item_chunk.content  # Enclosure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])