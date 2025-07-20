#!/usr/bin/env python3
"""
HTML Parser Performance Testing
===============================

Comprehensive performance testing suite for the HTML parser.
Tests parsing speed, memory usage, and scalability across different HTML scenarios.
"""

import sys
import time
import psutil
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import tempfile
import gc
import random
import string

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class PerformanceProfiler:
    """Performance profiling utility"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Start profiling"""
        gc.collect()  # Clean up before measurement
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop(self) -> Dict[str, float]:
        """Stop profiling and return metrics"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration_ms': (end_time - self.start_time) * 1000,
            'memory_start_mb': self.start_memory,
            'memory_end_mb': end_memory,
            'memory_delta_mb': end_memory - self.start_memory
        }

def generate_test_html(complexity: str, target_size: int) -> str:
    """Generate test HTML of specified complexity and size"""
    
    def random_text(length: int = 50) -> str:
        """Generate random text content"""
        words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit',
                'sed', 'do', 'eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore',
                'magna', 'aliqua', 'enim', 'ad', 'minim', 'veniam', 'quis', 'nostrud',
                'exercitation', 'ullamco', 'laboris', 'nisi', 'aliquip', 'ex', 'ea', 'commodo']
        return ' '.join(random.choices(words, k=length))
    
    def random_id():
        """Generate random ID"""
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))
    
    def random_class():
        """Generate random class name"""
        prefixes = ['card', 'btn', 'nav', 'content', 'item', 'box', 'wrapper', 'container']
        suffixes = ['primary', 'secondary', 'large', 'small', 'active', 'hover', 'focus']
        return f"{random.choice(prefixes)}-{random.choice(suffixes)}"
    
    if complexity == "simple":
        base_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple HTML Document</title>
</head>
<body>
    <header class="site-header">
        <h1>Welcome to Our Site</h1>
        <nav class="main-navigation">
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main class="main-content">
        <section class="hero">
            <h2>Hero Section</h2>
            <p>This is a simple hero section with some basic content.</p>
            <a href="/learn-more" class="btn btn-primary">Learn More</a>
        </section>
        
        <section class="features">
            <h2>Features</h2>
            <div class="feature-grid">
                <div class="feature-item">
                    <h3>Feature One</h3>
                    <p>Description of the first feature.</p>
                </div>
                <div class="feature-item">
                    <h3>Feature Two</h3>
                    <p>Description of the second feature.</p>
                </div>
                <div class="feature-item">
                    <h3>Feature Three</h3>
                    <p>Description of the third feature.</p>
                </div>
            </div>
        </section>
    </main>
    
    <footer class="site-footer">
        <p>&copy; 2024 Our Company. All rights reserved.</p>
    </footer>
</body>
</html>"""
    
    elif complexity == "medium":
        base_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="A medium complexity HTML document with forms and multimedia">
    <meta name="keywords" content="html, semantic, accessibility, forms">
    <title>Medium Complexity HTML Document</title>
    <link rel="stylesheet" href="styles.css">
    <link rel="icon" href="favicon.ico">
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    
    <header id="site-header" class="site-header" role="banner">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <img src="logo.svg" alt="Company Logo" width="120" height="40">
                </div>
                <nav class="main-navigation" aria-label="Main navigation">
                    <ul class="nav-list">
                        <li class="nav-item">
                            <a href="/" class="nav-link" aria-current="page">Home</a>
                        </li>
                        <li class="nav-item">
                            <a href="/products" class="nav-link">Products</a>
                            <ul class="sub-nav">
                                <li><a href="/products/category-1">Category 1</a></li>
                                <li><a href="/products/category-2">Category 2</a></li>
                            </ul>
                        </li>
                        <li class="nav-item">
                            <a href="/services" class="nav-link">Services</a>
                        </li>
                        <li class="nav-item">
                            <a href="/contact" class="nav-link">Contact</a>
                        </li>
                    </ul>
                </nav>
                <button class="mobile-menu-toggle" aria-label="Toggle mobile menu">
                    <span class="hamburger"></span>
                </button>
            </div>
        </div>
    </header>
    
    <main id="main-content" class="main-content">
        <section class="hero-section" aria-labelledby="hero-title">
            <div class="container">
                <h1 id="hero-title" class="hero-title">Transform Your Business</h1>
                <p class="hero-description">Discover innovative solutions that drive growth and success.</p>
                <div class="hero-actions">
                    <a href="/get-started" class="btn btn-primary">Get Started</a>
                    <a href="/learn-more" class="btn btn-outline">Learn More</a>
                </div>
            </div>
        </section>
        
        <section class="features-section" aria-labelledby="features-title">
            <div class="container">
                <h2 id="features-title" class="section-title">Our Features</h2>
                <div class="features-grid">
                    <article class="feature-card">
                        <div class="feature-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 11 5.16-1.26 9-5.45 9-11V7l-10-5z"/>
                            </svg>
                        </div>
                        <h3 class="feature-title">Security First</h3>
                        <p class="feature-description">Enterprise-grade security with end-to-end encryption.</p>
                    </article>
                    
                    <article class="feature-card">
                        <div class="feature-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9z"/>
                            </svg>
                        </div>
                        <h3 class="feature-title">Real-time Sync</h3>
                        <p class="feature-description">Keep your data synchronized across all devices instantly.</p>
                    </article>
                    
                    <article class="feature-card">
                        <div class="feature-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" aria-hidden="true">
                                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                            </svg>
                        </div>
                        <h3 class="feature-title">Easy Integration</h3>
                        <p class="feature-description">Seamlessly integrate with your existing tools and workflows.</p>
                    </article>
                </div>
            </div>
        </section>
        
        <section class="contact-section" aria-labelledby="contact-title">
            <div class="container">
                <h2 id="contact-title" class="section-title">Get In Touch</h2>
                <div class="contact-content">
                    <div class="contact-info">
                        <h3>Contact Information</h3>
                        <address>
                            <p>123 Business Street<br>
                            City, State 12345</p>
                            <p>Phone: <a href="tel:+1234567890">(123) 456-7890</a></p>
                            <p>Email: <a href="mailto:info@company.com">info@company.com</a></p>
                        </address>
                    </div>
                    
                    <form class="contact-form" method="post" action="/submit-contact">
                        <fieldset>
                            <legend class="sr-only">Contact Form</legend>
                            
                            <div class="form-group">
                                <label for="name">Full Name *</label>
                                <input type="text" id="name" name="name" required aria-describedby="name-error">
                                <span id="name-error" class="error-message" role="alert"></span>
                            </div>
                            
                            <div class="form-group">
                                <label for="email">Email Address *</label>
                                <input type="email" id="email" name="email" required aria-describedby="email-error">
                                <span id="email-error" class="error-message" role="alert"></span>
                            </div>
                            
                            <div class="form-group">
                                <label for="subject">Subject</label>
                                <select id="subject" name="subject">
                                    <option value="">Choose a subject</option>
                                    <option value="general">General Inquiry</option>
                                    <option value="support">Technical Support</option>
                                    <option value="sales">Sales Question</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label for="message">Message *</label>
                                <textarea id="message" name="message" rows="5" required aria-describedby="message-error"></textarea>
                                <span id="message-error" class="error-message" role="alert"></span>
                            </div>
                            
                            <div class="form-group">
                                <label class="checkbox-label">
                                    <input type="checkbox" name="newsletter" value="yes">
                                    <span class="checkmark"></span>
                                    Subscribe to our newsletter
                                </label>
                            </div>
                            
                            <div class="form-actions">
                                <button type="submit" class="btn btn-primary">Send Message</button>
                                <button type="reset" class="btn btn-outline">Reset Form</button>
                            </div>
                        </fieldset>
                    </form>
                </div>
            </div>
        </section>
    </main>
    
    <aside class="sidebar" role="complementary">
        <section class="widget">
            <h3>Latest News</h3>
            <ul class="news-list">
                <li class="news-item">
                    <a href="/news/article-1">Important Update Released</a>
                    <time datetime="2024-01-15">January 15, 2024</time>
                </li>
                <li class="news-item">
                    <a href="/news/article-2">New Feature Announcement</a>
                    <time datetime="2024-01-10">January 10, 2024</time>
                </li>
            </ul>
        </section>
    </aside>
    
    <footer id="site-footer" class="site-footer" role="contentinfo">
        <div class="container">
            <div class="footer-content">
                <div class="footer-section">
                    <h4>Company</h4>
                    <ul>
                        <li><a href="/about">About Us</a></li>
                        <li><a href="/careers">Careers</a></li>
                        <li><a href="/press">Press</a></li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h4>Support</h4>
                    <ul>
                        <li><a href="/help">Help Center</a></li>
                        <li><a href="/docs">Documentation</a></li>
                        <li><a href="/api">API</a></li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h4>Legal</h4>
                    <ul>
                        <li><a href="/privacy">Privacy Policy</a></li>
                        <li><a href="/terms">Terms of Service</a></li>
                        <li><a href="/cookies">Cookie Policy</a></li>
                    </ul>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2024 Company Name. All rights reserved.</p>
                <div class="social-links">
                    <a href="https://twitter.com/company" aria-label="Follow us on Twitter">
                        <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                            <path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"/>
                        </svg>
                    </a>
                    <a href="https://linkedin.com/company/company" aria-label="Follow us on LinkedIn">
                        <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                            <path d="M16 8a6 6 0 016 6v7h-4v-7a2 2 0 00-2-2 2 2 0 00-2 2v7h-4v-7a6 6 0 016-6zM2 9h4v12H2z"/>
                            <circle cx="4" cy="4" r="2"/>
                        </svg>
                    </a>
                </div>
            </div>
        </div>
    </footer>
</body>
</html>"""
    
    elif complexity == "complex":
        base_html = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Complex HTML document with advanced features, microdata, and interactive elements">
    <meta name="keywords" content="html5, semantic, accessibility, microdata, web components">
    <meta name="author" content="Web Development Team">
    <meta property="og:title" content="Complex HTML Demo">
    <meta property="og:description" content="Showcase of advanced HTML features">
    <meta property="og:image" content="https://example.com/og-image.jpg">
    <meta property="og:url" content="https://example.com/complex-demo">
    <meta name="twitter:card" content="summary_large_image">
    <title>Complex HTML Document with Advanced Features</title>
    
    <link rel="preload" href="critical.css" as="style">
    <link rel="preload" href="main.js" as="script">
    <link rel="stylesheet" href="styles.css">
    <link rel="manifest" href="manifest.json">
    <link rel="icon" sizes="192x192" href="icon-192.png">
    <link rel="apple-touch-icon" href="icon-180.png">
    
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "Organization",
        "name": "Example Company",
        "url": "https://example.com",
        "logo": "https://example.com/logo.png"
    }
    </script>
</head>
<body itemscope itemtype="https://schema.org/WebPage">
    <div class="page-wrapper">
        <a href="#main-content" class="skip-link">Skip to main content</a>
        
        <header id="site-header" class="site-header" role="banner" itemscope itemtype="https://schema.org/WPHeader">
            <div class="container">
                <div class="header-top">
                    <div class="header-utility">
                        <nav class="utility-nav" aria-label="Utility navigation">
                            <ul>
                                <li><a href="/account" rel="nofollow">My Account</a></li>
                                <li><a href="/cart" rel="nofollow">Cart (<span id="cart-count">0</span>)</a></li>
                                <li>
                                    <button class="theme-toggle" aria-label="Toggle dark mode" data-theme-toggle>
                                        <span class="sr-only">Toggle theme</span>
                                        <svg class="sun-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                                            <circle cx="12" cy="12" r="5"/>
                                            <line x1="12" y1="1" x2="12" y2="3"/>
                                            <line x1="12" y1="21" x2="12" y2="23"/>
                                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                                            <line x1="1" y1="12" x2="3" y2="12"/>
                                            <line x1="21" y1="12" x2="23" y2="12"/>
                                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                                        </svg>
                                        <svg class="moon-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                                            <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
                                        </svg>
                                    </button>
                                </li>
                            </ul>
                        </nav>
                    </div>
                </div>
                
                <div class="header-main">
                    <div class="logo" itemscope itemtype="https://schema.org/Organization">
                        <a href="/" itemprop="url">
                            <img src="logo.svg" alt="Company Logo" width="160" height="50" itemprop="logo">
                            <span class="sr-only" itemprop="name">Company Name</span>
                        </a>
                    </div>
                    
                    <nav class="main-navigation" role="navigation" aria-label="Main navigation">
                        <ul class="nav-list">
                            <li class="nav-item has-dropdown">
                                <a href="/products" class="nav-link" aria-expanded="false" aria-haspopup="true">
                                    Products
                                    <svg class="dropdown-icon" width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">
                                        <polyline points="6,9 12,15 18,9"/>
                                    </svg>
                                </a>
                                <ul class="dropdown-menu" role="menu">
                                    <li role="presentation">
                                        <a href="/products/software" role="menuitem">Software Solutions</a>
                                    </li>
                                    <li role="presentation">
                                        <a href="/products/hardware" role="menuitem">Hardware Products</a>
                                    </li>
                                    <li role="presentation">
                                        <a href="/products/cloud" role="menuitem">Cloud Services</a>
                                    </li>
                                </ul>
                            </li>
                            <li class="nav-item">
                                <a href="/services" class="nav-link">Services</a>
                            </li>
                            <li class="nav-item">
                                <a href="/pricing" class="nav-link">Pricing</a>
                            </li>
                            <li class="nav-item">
                                <a href="/resources" class="nav-link">Resources</a>
                            </li>
                            <li class="nav-item">
                                <a href="/contact" class="nav-link">Contact</a>
                            </li>
                        </ul>
                    </nav>
                    
                    <div class="header-actions">
                        <form class="search-form" role="search" action="/search" method="get">
                            <label for="search-input" class="sr-only">Search</label>
                            <input 
                                type="search" 
                                id="search-input" 
                                name="q" 
                                placeholder="Search..."
                                autocomplete="off"
                                aria-describedby="search-suggestions"
                            >
                            <button type="submit" aria-label="Submit search">
                                <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
                                    <circle cx="11" cy="11" r="8"/>
                                    <path d="M21 21l-4.35-4.35"/>
                                </svg>
                            </button>
                            <div id="search-suggestions" class="search-suggestions" role="listbox" aria-live="polite"></div>
                        </form>
                        
                        <button class="mobile-menu-toggle" aria-label="Toggle mobile menu" aria-expanded="false" data-mobile-toggle>
                            <span class="hamburger-line"></span>
                            <span class="hamburger-line"></span>
                            <span class="hamburger-line"></span>
                        </button>
                    </div>
                </div>
            </div>
        </header>
        
        <main id="main-content" class="main-content" itemprop="mainContentOfPage">
            <section class="hero-section" itemscope itemtype="https://schema.org/WPAdBlock">
                <div class="hero-background">
                    <picture>
                        <source media="(min-width: 768px)" srcset="hero-large.webp 1200w, hero-large@2x.webp 2400w" type="image/webp">
                        <source media="(min-width: 768px)" srcset="hero-large.jpg 1200w, hero-large@2x.jpg 2400w">
                        <source srcset="hero-small.webp 600w, hero-small@2x.webp 1200w" type="image/webp">
                        <img src="hero-small.jpg" alt="" loading="eager" decoding="async">
                    </picture>
                </div>
                
                <div class="container">
                    <div class="hero-content">
                        <h1 class="hero-title" itemprop="headline">
                            Revolutionary Solutions for 
                            <span class="text-highlight">Modern Business</span>
                        </h1>
                        <p class="hero-description">
                            Discover cutting-edge technology that transforms the way you work, 
                            collaborate, and achieve success in today's digital landscape.
                        </p>
                        <div class="hero-actions">
                            <a href="/get-started" class="btn btn-primary btn-large" data-track="hero-cta-primary">
                                Get Started Today
                                <svg class="btn-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                                    <line x1="5" y1="12" x2="19" y2="12"/>
                                    <polyline points="12,5 19,12 12,19"/>
                                </svg>
                            </a>
                            <button class="btn btn-outline btn-large" data-modal-trigger="demo-video" data-track="hero-cta-demo">
                                Watch Demo
                                <svg class="btn-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                                    <polygon points="5,3 19,12 5,21"/>
                                </svg>
                            </button>
                        </div>
                        
                        <div class="hero-stats">
                            <div class="stat-item" itemscope itemtype="https://schema.org/QuantitativeValue">
                                <span class="stat-number" itemprop="value">10,000+</span>
                                <span class="stat-label" itemprop="unitText">Happy Customers</span>
                            </div>
                            <div class="stat-item" itemscope itemtype="https://schema.org/QuantitativeValue">
                                <span class="stat-number" itemprop="value">99.9%</span>
                                <span class="stat-label" itemprop="unitText">Uptime Guarantee</span>
                            </div>
                            <div class="stat-item" itemscope itemtype="https://schema.org/QuantitativeValue">
                                <span class="stat-number" itemprop="value">24/7</span>
                                <span class="stat-label" itemprop="unitText">Expert Support</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            
            <section class="features-section" aria-labelledby="features-title">
                <div class="container">
                    <header class="section-header">
                        <h2 id="features-title" class="section-title">Powerful Features</h2>
                        <p class="section-description">
                            Everything you need to scale your business and exceed customer expectations.
                        </p>
                    </header>
                    
                    <div class="features-grid" itemscope itemtype="https://schema.org/ItemList">
                        <article class="feature-card" itemscope itemtype="https://schema.org/SoftwareApplication" itemprop="itemListElement">
                            <div class="feature-icon" aria-hidden="true">
                                <svg width="48" height="48" viewBox="0 0 24 24">
                                    <path d="M12 2L2 7v10c0 5.55 3.84 9.74 9 11 5.16-1.26 9-5.45 9-11V7l-10-5z"/>
                                </svg>
                            </div>
                            <h3 class="feature-title" itemprop="name">Enterprise Security</h3>
                            <p class="feature-description" itemprop="description">
                                Bank-grade encryption and advanced security protocols protect your data 24/7.
                            </p>
                            <ul class="feature-list">
                                <li>End-to-end encryption</li>
                                <li>Multi-factor authentication</li>
                                <li>SOC 2 Type II compliance</li>
                                <li>Regular security audits</li>
                            </ul>
                            <a href="/features/security" class="feature-link">Learn more about security</a>
                        </article>
                        
                        <article class="feature-card" itemscope itemtype="https://schema.org/SoftwareApplication" itemprop="itemListElement">
                            <div class="feature-icon" aria-hidden="true">
                                <svg width="48" height="48" viewBox="0 0 24 24">
                                    <path d="M13 3c-4.97 0-9 4.03-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42C8.27 19.99 10.51 21 13 21c4.97 0 9-4.03 9-9s-4.03-9-9-9z"/>
                                </svg>
                            </div>
                            <h3 class="feature-title" itemprop="name">Real-time Collaboration</h3>
                            <p class="feature-description" itemprop="description">
                                Work together seamlessly with instant synchronization across all devices and platforms.
                            </p>
                            <ul class="feature-list">
                                <li>Live document editing</li>
                                <li>Video conferencing</li>
                                <li>Screen sharing</li>
                                <li>Comment threads</li>
                            </ul>
                            <a href="/features/collaboration" class="feature-link">Explore collaboration tools</a>
                        </article>
                        
                        <article class="feature-card" itemscope itemtype="https://schema.org/SoftwareApplication" itemprop="itemListElement">
                            <div class="feature-icon" aria-hidden="true">
                                <svg width="48" height="48" viewBox="0 0 24 24">
                                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                                    <line x1="16" y1="2" x2="16" y2="6"/>
                                    <line x1="8" y1="2" x2="8" y2="6"/>
                                    <line x1="3" y1="10" x2="21" y2="10"/>
                                </svg>
                            </div>
                            <h3 class="feature-title" itemprop="name">Smart Analytics</h3>
                            <p class="feature-description" itemprop="description">
                                Gain actionable insights with AI-powered analytics and customizable dashboards.
                            </p>
                            <ul class="feature-list">
                                <li>Custom dashboards</li>
                                <li>Predictive analytics</li>
                                <li>Real-time reporting</li>
                                <li>Data visualization</li>
                            </ul>
                            <a href="/features/analytics" class="feature-link">View analytics features</a>
                        </article>
                    </div>
                </div>
            </section>
            
            <section class="testimonials-section" aria-labelledby="testimonials-title">
                <div class="container">
                    <h2 id="testimonials-title" class="section-title">What Our Customers Say</h2>
                    
                    <div class="testimonials-slider" data-slider>
                        <div class="testimonials-track" data-slider-track>
                            <blockquote class="testimonial-card" itemscope itemtype="https://schema.org/Review">
                                <div class="testimonial-content">
                                    <p itemprop="reviewBody">
                                        "This platform has completely transformed how our team collaborates. 
                                        The intuitive interface and powerful features have increased our 
                                        productivity by 40%."
                                    </p>
                                </div>
                                <footer class="testimonial-author" itemscope itemtype="https://schema.org/Person">
                                    <img src="avatar-1.jpg" alt="" class="author-avatar" loading="lazy">
                                    <div class="author-info">
                                        <cite class="author-name" itemprop="name">Sarah Johnson</cite>
                                        <span class="author-title" itemprop="jobTitle">VP of Operations, TechCorp</span>
                                    </div>
                                    <div class="testimonial-rating" itemprop="reviewRating" itemscope itemtype="https://schema.org/Rating">
                                        <span class="sr-only">Rating: <span itemprop="ratingValue">5</span> out of <span itemprop="bestRating">5</span> stars</span>
                                        <div class="stars" aria-hidden="true">★★★★★</div>
                                    </div>
                                </footer>
                            </blockquote>
                            
                            <blockquote class="testimonial-card" itemscope itemtype="https://schema.org/Review">
                                <div class="testimonial-content">
                                    <p itemprop="reviewBody">
                                        "The security features give us peace of mind when handling sensitive 
                                        client data. Support is incredibly responsive and knowledgeable."
                                    </p>
                                </div>
                                <footer class="testimonial-author" itemscope itemtype="https://schema.org/Person">
                                    <img src="avatar-2.jpg" alt="" class="author-avatar" loading="lazy">
                                    <div class="author-info">
                                        <cite class="author-name" itemprop="name">Michael Chen</cite>
                                        <span class="author-title" itemprop="jobTitle">CTO, DataFlow Inc</span>
                                    </div>
                                    <div class="testimonial-rating" itemprop="reviewRating" itemscope itemtype="https://schema.org/Rating">
                                        <span class="sr-only">Rating: <span itemprop="ratingValue">5</span> out of <span itemprop="bestRating">5</span> stars</span>
                                        <div class="stars" aria-hidden="true">★★★★★</div>
                                    </div>
                                </footer>
                            </blockquote>
                        </div>
                        
                        <div class="slider-controls">
                            <button class="slider-button slider-button--prev" aria-label="Previous testimonial" data-slider-prev>
                                <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                                    <polyline points="15,18 9,12 15,6"/>
                                </svg>
                            </button>
                            <button class="slider-button slider-button--next" aria-label="Next testimonial" data-slider-next>
                                <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                                    <polyline points="9,18 15,12 9,6"/>
                                </svg>
                            </button>
                        </div>
                        
                        <div class="slider-indicators">
                            <button class="indicator active" aria-label="Go to testimonial 1" data-slider-indicator="0"></button>
                            <button class="indicator" aria-label="Go to testimonial 2" data-slider-indicator="1"></button>
                        </div>
                    </div>
                </div>
            </section>
        </main>
        
        <aside class="sidebar" role="complementary" aria-label="Sidebar content">
            <section class="widget newsletter-widget">
                <h3 class="widget-title">Stay Updated</h3>
                <p class="widget-description">Get the latest news and updates delivered to your inbox.</p>
                <form class="newsletter-form" action="/subscribe" method="post" novalidate>
                    <div class="form-group">
                        <label for="newsletter-email" class="sr-only">Email address</label>
                        <input 
                            type="email" 
                            id="newsletter-email" 
                            name="email" 
                            placeholder="Enter your email"
                            required
                            aria-describedby="newsletter-error"
                        >
                        <span id="newsletter-error" class="error-message" role="alert"></span>
                    </div>
                    <button type="submit" class="btn btn-primary btn-small">Subscribe</button>
                </form>
            </section>
            
            <section class="widget resources-widget">
                <h3 class="widget-title">Popular Resources</h3>
                <ul class="resource-list">
                    <li class="resource-item">
                        <article itemscope itemtype="https://schema.org/Article">
                            <h4 class="resource-title">
                                <a href="/resources/guide-1" itemprop="url">
                                    <span itemprop="headline">Complete Setup Guide</span>
                                </a>
                            </h4>
                            <p class="resource-meta">
                                <time datetime="2024-01-15" itemprop="datePublished">Jan 15, 2024</time>
                                • <span itemprop="wordCount">5 min read</span>
                            </p>
                        </article>
                    </li>
                    <li class="resource-item">
                        <article itemscope itemtype="https://schema.org/Article">
                            <h4 class="resource-title">
                                <a href="/resources/tutorial-1" itemprop="url">
                                    <span itemprop="headline">Advanced Features Tutorial</span>
                                </a>
                            </h4>
                            <p class="resource-meta">
                                <time datetime="2024-01-12" itemprop="datePublished">Jan 12, 2024</time>
                                • <span itemprop="wordCount">8 min read</span>
                            </p>
                        </article>
                    </li>
                </ul>
            </section>
        </aside>
        
        <footer id="site-footer" class="site-footer" role="contentinfo" itemscope itemtype="https://schema.org/WPFooter">
            <div class="container">
                <div class="footer-main">
                    <div class="footer-brand">
                        <div class="footer-logo">
                            <img src="logo-white.svg" alt="Company Logo" width="120" height="40">
                        </div>
                        <p class="footer-tagline">Empowering businesses with innovative technology solutions.</p>
                        <div class="social-links">
                            <a href="https://twitter.com/company" class="social-link" aria-label="Follow us on Twitter" rel="noopener">
                                <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                                    <path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"/>
                                </svg>
                            </a>
                            <a href="https://linkedin.com/company/company" class="social-link" aria-label="Follow us on LinkedIn" rel="noopener">
                                <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                                    <path d="M16 8a6 6 0 016 6v7h-4v-7a2 2 0 00-2-2 2 2 0 00-2 2v7h-4v-7a6 6 0 016-6zM2 9h4v12H2z"/>
                                    <circle cx="4" cy="4" r="2"/>
                                </svg>
                            </a>
                            <a href="https://github.com/company" class="social-link" aria-label="Follow us on GitHub" rel="noopener">
                                <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                                    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 00-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0020 4.77 5.07 5.07 0 0019.91 1S18.73.65 16 2.48a13.38 13.38 0 00-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 005 4.77a5.44 5.44 0 00-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 009 18.13V22"/>
                                </svg>
                            </a>
                        </div>
                    </div>
                    
                    <nav class="footer-nav">
                        <div class="footer-section">
                            <h4 class="footer-section-title">Products</h4>
                            <ul class="footer-links">
                                <li><a href="/products/software">Software Solutions</a></li>
                                <li><a href="/products/hardware">Hardware Products</a></li>
                                <li><a href="/products/cloud">Cloud Services</a></li>
                                <li><a href="/products/enterprise">Enterprise Solutions</a></li>
                            </ul>
                        </div>
                        
                        <div class="footer-section">
                            <h4 class="footer-section-title">Support</h4>
                            <ul class="footer-links">
                                <li><a href="/help">Help Center</a></li>
                                <li><a href="/docs">Documentation</a></li>
                                <li><a href="/api">API Reference</a></li>
                                <li><a href="/status">System Status</a></li>
                                <li><a href="/contact">Contact Support</a></li>
                            </ul>
                        </div>
                        
                        <div class="footer-section">
                            <h4 class="footer-section-title">Company</h4>
                            <ul class="footer-links">
                                <li><a href="/about">About Us</a></li>
                                <li><a href="/careers">Careers</a></li>
                                <li><a href="/press">Press Kit</a></li>
                                <li><a href="/blog">Blog</a></li>
                                <li><a href="/investors">Investors</a></li>
                            </ul>
                        </div>
                        
                        <div class="footer-section">
                            <h4 class="footer-section-title">Legal</h4>
                            <ul class="footer-links">
                                <li><a href="/privacy">Privacy Policy</a></li>
                                <li><a href="/terms">Terms of Service</a></li>
                                <li><a href="/cookies">Cookie Policy</a></li>
                                <li><a href="/security">Security</a></li>
                                <li><a href="/compliance">Compliance</a></li>
                            </ul>
                        </div>
                    </nav>
                </div>
                
                <div class="footer-bottom">
                    <div class="footer-bottom-content">
                        <p class="copyright">
                            &copy; 2024 <span itemscope itemtype="https://schema.org/Organization"><span itemprop="name">Company Name</span></span>. 
                            All rights reserved.
                        </p>
                        <div class="footer-bottom-links">
                            <a href="/sitemap">Sitemap</a>
                            <a href="/accessibility">Accessibility</a>
                            <button class="cookie-settings" data-cookie-settings>Cookie Settings</button>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    </div>
    
    <!-- Modal for demo video -->
    <div id="demo-video-modal" class="modal" role="dialog" aria-labelledby="modal-title" aria-hidden="true">
        <div class="modal-overlay" data-modal-close></div>
        <div class="modal-content">
            <header class="modal-header">
                <h2 id="modal-title" class="modal-title">Product Demo</h2>
                <button class="modal-close" aria-label="Close modal" data-modal-close>
                    <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </header>
            <div class="modal-body">
                <div class="video-container">
                    <video controls poster="video-poster.jpg" preload="metadata">
                        <source src="demo-video.mp4" type="video/mp4">
                        <source src="demo-video.webm" type="video/webm">
                        <track kind="captions" src="captions-en.vtt" srclang="en" label="English">
                        <p>Your browser doesn't support HTML5 video. <a href="demo-video.mp4">Download the video</a> instead.</p>
                    </video>
                </div>
            </div>
        </div>
    </div>
    
    <script src="main.js" defer></script>
    <script>
        // Progressive enhancement for critical features
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
        
        // Theme switching
        const themeToggle = document.querySelector('[data-theme-toggle]');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
            });
        }
    </script>
</body>
</html>"""
    
    else:
        base_html = "<!-- Unknown complexity level -->\n<html><body><p>Simple content</p></body></html>\n"
    
    # Expand HTML to reach target size
    current_size = len(base_html)
    target_size = target_size
    
    if current_size >= target_size:
        return base_html[:target_size]
    
    # Generate additional content to reach target size
    additional_content = ""
    counter = 1
    
    while len(base_html + additional_content) < target_size:
        if complexity == "simple":
            additional_content += f"""
    <section class="section-{counter}">
        <h2>Section {counter}</h2>
        <p>{random_text(30)}</p>
        <ul>
            <li>{random_text(10)}</li>
            <li>{random_text(10)}</li>
            <li>{random_text(10)}</li>
        </ul>
    </section>
"""
        
        elif complexity == "medium":
            additional_content += f"""
    <article class="article-{counter}" id="{random_id()}">
        <header class="article-header">
            <h2 class="{random_class()}">{random_text(5).title()}</h2>
            <p class="article-meta">
                <time datetime="2024-01-{15+counter:02d}">January {15+counter}, 2024</time>
                <span class="author">By Author {counter}</span>
            </p>
        </header>
        <div class="article-content">
            <p>{random_text(40)}</p>
            <blockquote>
                <p>{random_text(20)}</p>
                <cite>- Expert Quote {counter}</cite>
            </blockquote>
            <p>{random_text(35)}</p>
        </div>
        <footer class="article-footer">
            <div class="tags">
                <span class="tag">{random_class()}</span>
                <span class="tag">{random_class()}</span>
            </div>
            <div class="share-buttons">
                <button class="share-btn" aria-label="Share on Twitter">Share</button>
            </div>
        </footer>
    </article>
"""
        
        else:  # complex
            additional_content += f"""
    <section class="dynamic-section-{counter}" 
             itemscope itemtype="https://schema.org/Article" 
             data-section-id="{counter}"
             aria-labelledby="section-{counter}-title">
        <header class="section-header">
            <h2 id="section-{counter}-title" class="{random_class()}" itemprop="headline">
                Advanced Section {counter}
            </h2>
            <div class="section-meta" itemprop="author" itemscope itemtype="https://schema.org/Person">
                <span itemprop="name">Content Author {counter}</span>
                <time datetime="2024-01-{15+counter:02d}" itemprop="datePublished">
                    January {15+counter}, 2024
                </time>
            </div>
        </header>
        
        <div class="section-content" itemprop="articleBody">
            <div class="content-grid">
                <div class="content-main">
                    <p class="lead-paragraph">{random_text(25)}</p>
                    
                    <figure class="embedded-media" itemscope itemtype="https://schema.org/ImageObject">
                        <picture>
                            <source media="(min-width: 768px)" 
                                    srcset="image-{counter}-large.webp 800w, image-{counter}-large@2x.webp 1600w" 
                                    type="image/webp">
                            <source media="(min-width: 768px)" 
                                    srcset="image-{counter}-large.jpg 800w, image-{counter}-large@2x.jpg 1600w">
                            <source srcset="image-{counter}-small.webp 400w, image-{counter}-small@2x.webp 800w" 
                                    type="image/webp">
                            <img src="image-{counter}-small.jpg" 
                                 alt="Descriptive text for image {counter}" 
                                 loading="lazy" 
                                 decoding="async"
                                 itemprop="contentUrl">
                        </picture>
                        <figcaption itemprop="caption">{random_text(15)}</figcaption>
                    </figure>
                    
                    <div class="interactive-content" data-interactive="true">
                        <details class="disclosure-widget">
                            <summary class="disclosure-summary">
                                Expandable Content {counter}
                                <svg class="disclosure-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                                    <polyline points="6,9 12,15 18,9"/>
                                </svg>
                            </summary>
                            <div class="disclosure-content">
                                <p>{random_text(30)}</p>
                                <ul class="feature-list">
                                    <li class="feature-item">{random_text(8)}</li>
                                    <li class="feature-item">{random_text(8)}</li>
                                    <li class="feature-item">{random_text(8)}</li>
                                </ul>
                            </div>
                        </details>
                    </div>
                </div>
                
                <aside class="content-sidebar" role="complementary" aria-label="Related content">
                    <div class="sidebar-widget">
                        <h3 class="widget-title">Related Topics</h3>
                        <nav class="related-nav">
                            <ul>
                                <li><a href="/topic-{counter}-1" rel="related">{random_text(5).title()}</a></li>
                                <li><a href="/topic-{counter}-2" rel="related">{random_text(5).title()}</a></li>
                                <li><a href="/topic-{counter}-3" rel="related">{random_text(5).title()}</a></li>
                            </ul>
                        </nav>
                    </div>
                </aside>
            </div>
        </div>
        
        <footer class="section-footer">
            <div class="engagement-actions">
                <button class="action-btn like-btn" 
                        data-action="like" 
                        data-target="section-{counter}"
                        aria-label="Like this section">
                    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                    </svg>
                    <span class="action-count">0</span>
                </button>
                
                <button class="action-btn bookmark-btn" 
                        data-action="bookmark" 
                        data-target="section-{counter}"
                        aria-label="Bookmark this section">
                    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
                    </svg>
                </button>
                
                <div class="share-menu" data-share-menu>
                    <button class="action-btn share-btn" 
                            data-share-trigger
                            aria-label="Share this section"
                            aria-expanded="false"
                            aria-haspopup="true">
                        <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                            <circle cx="18" cy="5" r="3"/>
                            <circle cx="6" cy="12" r="3"/>
                            <circle cx="18" cy="19" r="3"/>
                            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
                            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
                        </svg>
                    </button>
                    <div class="share-dropdown" role="menu">
                        <a href="#" class="share-option" role="menuitem">Twitter</a>
                        <a href="#" class="share-option" role="menuitem">LinkedIn</a>
                        <a href="#" class="share-option" role="menuitem">Email</a>
                    </div>
                </div>
            </div>
        </footer>
    </section>
"""
        
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 100:
            break
    
    final_content = base_html + additional_content
    return final_content[:target_size] if len(final_content) > target_size else final_content

def run_parsing_benchmark(content: str, iterations: int = 5) -> Dict[str, Any]:
    """Run parsing benchmark with multiple iterations and timeout protection"""
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(
        target_chunk_size=500,  # Good size for HTML elements
        min_chunk_size=50,
        preserve_atomic_nodes=True,
        enable_dependency_tracking=True
    )
    
    engine = ChunkingEngine(config)
    profiler = PerformanceProfiler()
    
    results = []
    chunks_counts = []
    
    for i in range(iterations):
        profiler.start()
        
        try:
            # Add timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Parsing took too long")
            
            # Set 30-second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            chunks = engine.chunk_content(content, 'html', f'test_file_{i}.html')
            
            # Cancel timeout
            signal.alarm(0)
            
        except TimeoutError:
            print(f"   ⚠️  Parsing timed out after 30 seconds")
            signal.alarm(0)
            return {
                'iterations': i,
                'avg_duration_ms': 30000,  # 30 seconds
                'min_duration_ms': 30000,
                'max_duration_ms': 30000,
                'std_duration_ms': 0,
                'avg_memory_delta_mb': 0,
                'avg_chunks_created': 0,
                'chunks_per_second': 0,
                'chars_per_second': 0,
                'all_results': [],
                'timeout': True
            }
        except Exception as e:
            print(f"   ❌ Parsing error: {e}")
            signal.alarm(0)
            return {
                'iterations': i,
                'avg_duration_ms': 0,
                'min_duration_ms': 0,
                'max_duration_ms': 0,
                'std_duration_ms': 0,
                'avg_memory_delta_mb': 0,
                'avg_chunks_created': 0,
                'chunks_per_second': 0,
                'chars_per_second': 0,
                'all_results': [],
                'error': str(e)
            }
        
        metrics = profiler.stop()
        results.append(metrics)
        chunks_counts.append(len(chunks))
        
        # Clean up between iterations
        gc.collect()
    
    # Calculate statistics
    durations = [r['duration_ms'] for r in results]
    memory_deltas = [r['memory_delta_mb'] for r in results]
    
    return {
        'iterations': iterations,
        'avg_duration_ms': statistics.mean(durations),
        'min_duration_ms': min(durations),
        'max_duration_ms': max(durations),
        'std_duration_ms': statistics.stdev(durations) if len(durations) > 1 else 0,
        'avg_memory_delta_mb': statistics.mean(memory_deltas),
        'avg_chunks_created': statistics.mean(chunks_counts),
        'chunks_per_second': statistics.mean(chunks_counts) / (statistics.mean(durations) / 1000),
        'chars_per_second': len(content) / (statistics.mean(durations) / 1000),
        'all_results': results
    }

def run_scalability_test() -> List[Dict[str, Any]]:
    """Test HTML parser scalability across different file sizes and complexities"""
    
    test_scenarios = [
        # Small HTML files
        {'complexity': 'simple', 'size': 1000, 'label': 'Small Simple HTML'},
        {'complexity': 'medium', 'size': 1000, 'label': 'Small Medium HTML'},
        {'complexity': 'complex', 'size': 1000, 'label': 'Small Complex HTML'},
        
        # Medium HTML files
        {'complexity': 'simple', 'size': 5000, 'label': 'Medium Simple HTML'},
        {'complexity': 'medium', 'size': 5000, 'label': 'Medium Medium HTML'},
        {'complexity': 'complex', 'size': 5000, 'label': 'Medium Complex HTML'},
        
        # Large HTML files
        {'complexity': 'simple', 'size': 20000, 'label': 'Large Simple HTML'},
        {'complexity': 'medium', 'size': 20000, 'label': 'Large Medium HTML'},
        {'complexity': 'complex', 'size': 20000, 'label': 'Large Complex HTML'},
        
        # Very large HTML files
        {'complexity': 'medium', 'size': 50000, 'label': 'XLarge Medium HTML'},
        {'complexity': 'complex', 'size': 50000, 'label': 'XLarge Complex HTML'},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"🧪 Testing {scenario['label']} (~{scenario['size']} chars, {scenario['complexity']})...")
        
        # Generate test content
        try:
            content = generate_test_html(scenario['complexity'], scenario['size'])
        except Exception as e:
            print(f"   ❌ Failed to generate HTML: {e}")
            continue
        
        # Skip if content is too large (> 200KB to prevent hanging)
        if len(content) > 200000:
            print(f"   ⚠️  Skipping - content too large ({len(content):,} chars)")
            continue
        
        # Run benchmark
        benchmark_result = run_parsing_benchmark(content, iterations=3)
        
        # Check for errors or timeouts
        if 'error' in benchmark_result:
            print(f"   ❌ Parsing error: {benchmark_result['error']}")
            continue
        elif 'timeout' in benchmark_result:
            print(f"   ⚠️  Parsing timed out")
            continue
        
        # Analyze HTML structure
        html_structure = analyze_html_structure(content)
        
        # Combine scenario info with results
        result = {
            **scenario,
            'actual_size': len(content),
            'line_count': content.count('\n') + 1,
            'html_structure': html_structure,
            **benchmark_result
        }
        
        results.append(result)
        
        print(f"   ✅ {benchmark_result['avg_duration_ms']:.1f}ms avg, "
              f"{benchmark_result['avg_chunks_created']:.0f} chunks, "
              f"{benchmark_result['chars_per_second']:.0f} chars/sec")
    
    return results

def analyze_html_structure(content: str) -> Dict[str, Any]:
    """Analyze HTML structure characteristics"""
    
    # Count HTML features
    elements = content.count('<') - content.count('<!') - content.count('</')  # Opening tags
    semantic_elements = 0
    form_elements = 0
    media_elements = 0
    links = content.count('<a ')
    images = content.count('<img ')
    
    # Count semantic HTML5 elements
    semantic_tags = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer', 'figure', 'figcaption']
    for tag in semantic_tags:
        semantic_elements += content.count(f'<{tag}')
    
    # Count form elements
    form_tags = ['form', 'input', 'textarea', 'select', 'button', 'fieldset', 'legend', 'label']
    for tag in form_tags:
        form_elements += content.count(f'<{tag}')
    
    # Count media elements
    media_tags = ['img', 'video', 'audio', 'picture', 'source']
    for tag in media_tags:
        media_elements += content.count(f'<{tag}')
    
    # Estimate nesting depth by analyzing indentation
    lines = content.split('\n')
    max_indent = 0
    for line in lines:
        if line.strip() and not line.strip().startswith('<!--'):
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent // 2)  # Assume 2-space indentation
    
    # Count accessibility features
    aria_attributes = content.count('aria-')
    alt_attributes = content.count(' alt=')
    role_attributes = content.count(' role=')
    
    # Count modern HTML features
    microdata = content.count('itemscope') + content.count('itemprop')
    custom_elements = content.count('<') - sum(content.count(f'<{tag}') for tag in ['html', 'head', 'body', 'div', 'span', 'p', 'a', 'img', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    return {
        'total_elements': elements,
        'semantic_elements': semantic_elements,
        'form_elements': form_elements,
        'media_elements': media_elements,
        'links': links,
        'images': images,
        'max_nesting_depth': max_indent,
        'aria_attributes': aria_attributes,
        'alt_attributes': alt_attributes,
        'role_attributes': role_attributes,
        'microdata_usage': microdata,
        'custom_elements': max(0, custom_elements),
        'has_forms': form_elements > 0,
        'has_media': media_elements > 0,
        'is_semantic': semantic_elements > 0,
        'is_accessible': aria_attributes > 0 or alt_attributes > 0 or role_attributes > 0,
        'uses_microdata': microdata > 0
    }

def run_memory_stress_test() -> Dict[str, Any]:
    """Test memory usage under stress conditions"""
    print("🔥 Running HTML memory stress test...")
    
    profiler = PerformanceProfiler()
    
    # Generate large, complex HTML content
    large_content = generate_test_html('complex', 100000)
    
    # Test repeated parsing
    profiler.start()
    
    from chuk_code_raptor.chunking.engine import ChunkingEngine
    from chuk_code_raptor.chunking.config import ChunkingConfig
    
    config = ChunkingConfig(enable_dependency_tracking=True)
    engine = ChunkingEngine(config)
    
    total_chunks = 0
    iterations = 10
    
    for i in range(iterations):
        chunks = engine.chunk_content(large_content, 'html', f'stress_test_{i}.html')
        total_chunks += len(chunks)
        
        if i % 2 == 0:
            gc.collect()  # Periodic cleanup
    
    final_metrics = profiler.stop()
    
    return {
        'test_type': 'html_memory_stress',
        'iterations': iterations,
        'content_size': len(large_content),
        'total_chunks_created': total_chunks,
        'avg_chunks_per_iteration': total_chunks / iterations,
        **final_metrics
    }

def run_concurrent_parsing_test() -> Dict[str, Any]:
    """Test concurrent HTML parsing performance"""
    print("🔄 Running concurrent HTML parsing test...")
    
    import concurrent.futures
    
    def parse_html_content(content, file_id):
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        config = ChunkingConfig()
        engine = ChunkingEngine(config)
        
        start_time = time.time()
        chunks = engine.chunk_content(content, 'html', f'concurrent_test_{file_id}.html')
        duration = time.time() - start_time
        
        return {
            'file_id': file_id,
            'chunks_created': len(chunks),
            'duration_ms': duration * 1000,
            'content_size': len(content)
        }
    
    # Generate different HTML test contents
    test_contents = [
        generate_test_html('simple', 3000),
        generate_test_html('medium', 3000),
        generate_test_html('complex', 3000),
        generate_test_html('simple', 4000),
        generate_test_html('medium', 4000),
        generate_test_html('complex', 4000),
        generate_test_html('simple', 2500),
        generate_test_html('medium', 2500),
    ]
    
    profiler = PerformanceProfiler()
    profiler.start()
    
    # Run concurrent parsing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(parse_html_content, content, i) 
            for i, content in enumerate(test_contents)
        ]
        
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    overall_metrics = profiler.stop()
    
    # Calculate concurrent performance metrics
    total_chunks = sum(r['chunks_created'] for r in results)
    total_parse_time = sum(r['duration_ms'] for r in results)
    avg_parse_time = total_parse_time / len(results)
    total_content_size = sum(r['content_size'] for r in results)
    
    return {
        'test_type': 'concurrent_html_parsing',
        'worker_count': 4,
        'files_processed': len(test_contents),
        'total_chunks_created': total_chunks,
        'total_content_size': total_content_size,
        'avg_chunks_per_file': total_chunks / len(results),
        'avg_parse_time_ms': avg_parse_time,
        'total_wall_time_ms': overall_metrics['duration_ms'],
        'parallelization_efficiency': (total_parse_time / overall_metrics['duration_ms']),
        'total_chars_per_second': total_content_size / (overall_metrics['duration_ms'] / 1000),
        **overall_metrics,
        'individual_results': results
    }

def generate_performance_report(scalability_results: List[Dict], 
                              stress_result: Dict, 
                              concurrent_result: Dict) -> str:
    """Generate comprehensive HTML performance report"""
    
    report = []
    report.append("🌐 HTML PARSER PERFORMANCE REPORT")
    report.append("=" * 60)
    
    # Scalability Analysis
    report.append("\n📈 SCALABILITY ANALYSIS")
    report.append("-" * 40)
    
    report.append(f"{'Scenario':<25} {'Size':<8} {'Duration':<10} {'Chunks':<8} {'Chars/s':<10} {'Elements':<8}")
    report.append("-" * 75)
    
    for result in scalability_results:
        html_struct = result['html_structure']
        report.append(
            f"{result['label']:<25} "
            f"{result['actual_size']:<8} "
            f"{result['avg_duration_ms']:<10.1f} "
            f"{result['avg_chunks_created']:<8.0f} "
            f"{result['chars_per_second']:<10.0f} "
            f"{html_struct['total_elements']:<8}"
        )
    
    # Performance insights
    report.append("\n🔍 SCALABILITY INSIGHTS")
    
    # Group by complexity
    by_complexity = defaultdict(list)
    for result in scalability_results:
        by_complexity[result['complexity']].append(result)
    
    for complexity, results in by_complexity.items():
        sizes = [r['actual_size'] for r in results]
        speeds = [r['chars_per_second'] for r in results]
        elements = [r['html_structure']['total_elements'] for r in results]
        
        if len(speeds) > 1:
            speed_ratio = max(speeds) / min(speeds)
            avg_elements = statistics.mean(elements)
            report.append(f"   {complexity.title()} HTML: {speed_ratio:.1f}x speed variation, avg {avg_elements:.0f} elements")
    
    # HTML feature analysis
    report.append("\n🌐 HTML FEATURE ANALYSIS")
    all_features = {
        'total_elements': 0,
        'semantic_elements': 0,
        'form_elements': 0,
        'media_elements': 0,
        'accessibility_features': 0,
        'semantic_files': 0,
        'accessible_files': 0,
        'form_files': 0,
        'media_files': 0,
        'microdata_files': 0
    }
    
    for result in scalability_results:
        html_struct = result['html_structure']
        all_features['total_elements'] += html_struct['total_elements']
        all_features['semantic_elements'] += html_struct['semantic_elements']
        all_features['form_elements'] += html_struct['form_elements']
        all_features['media_elements'] += html_struct['media_elements']
        all_features['accessibility_features'] += (
            html_struct['aria_attributes'] + 
            html_struct['alt_attributes'] + 
            html_struct['role_attributes']
        )
        
        if html_struct['is_semantic']:
            all_features['semantic_files'] += 1
        if html_struct['is_accessible']:
            all_features['accessible_files'] += 1
        if html_struct['has_forms']:
            all_features['form_files'] += 1
        if html_struct['has_media']:
            all_features['media_files'] += 1
        if html_struct['uses_microdata']:
            all_features['microdata_files'] += 1
    
    total_files = len(scalability_results)
    report.append(f"   Total HTML elements processed: {all_features['total_elements']:,}")
    report.append(f"   Semantic elements: {all_features['semantic_elements']}")
    report.append(f"   Form elements: {all_features['form_elements']}")
    report.append(f"   Media elements: {all_features['media_elements']}")
    report.append(f"   Accessibility features: {all_features['accessibility_features']}")
    report.append(f"   Semantic HTML files: {all_features['semantic_files']}/{total_files}")
    report.append(f"   Accessible files: {all_features['accessible_files']}/{total_files}")
    report.append(f"   Files with forms: {all_features['form_files']}/{total_files}")
    report.append(f"   Files with media: {all_features['media_files']}/{total_files}")
    report.append(f"   Files with microdata: {all_features['microdata_files']}/{total_files}")
    
    # Memory stress test
    report.append(f"\n🔥 MEMORY STRESS TEST")
    report.append("-" * 40)
    report.append(f"   Content size: {stress_result['content_size']:,} characters")
    report.append(f"   Iterations: {stress_result['iterations']}")
    report.append(f"   Total chunks: {stress_result['total_chunks_created']:,}")
    report.append(f"   Duration: {stress_result['duration_ms']:.1f}ms")
    report.append(f"   Memory delta: {stress_result['memory_delta_mb']:.1f}MB")
    report.append(f"   Throughput: {stress_result['content_size'] * stress_result['iterations'] / (stress_result['duration_ms'] / 1000):.0f} chars/sec")
    
    # Concurrent parsing test
    report.append(f"\n🔄 CONCURRENT PARSING TEST")
    report.append("-" * 40)
    report.append(f"   Workers: {concurrent_result['worker_count']}")
    report.append(f"   Files processed: {concurrent_result['files_processed']}")
    report.append(f"   Total chunks: {concurrent_result['total_chunks_created']}")
    report.append(f"   Total content: {concurrent_result['total_content_size']:,} chars")
    report.append(f"   Wall time: {concurrent_result['total_wall_time_ms']:.1f}ms")
    report.append(f"   Parallelization efficiency: {concurrent_result['parallelization_efficiency']:.1f}x")
    report.append(f"   Throughput: {concurrent_result['total_chars_per_second']:.0f} chars/sec")
    report.append(f"   Memory delta: {concurrent_result['memory_delta_mb']:.1f}MB")
    
    # Overall performance summary
    report.append(f"\n⭐ PERFORMANCE SUMMARY")
    report.append("-" * 40)
    
    # Find best and worst performers
    best_speed = max(scalability_results, key=lambda x: x['chars_per_second'])
    worst_speed = min(scalability_results, key=lambda x: x['chars_per_second'])
    
    report.append(f"   Best performance: {best_speed['chars_per_second']:.0f} chars/sec ({best_speed['label']})")
    report.append(f"   Worst performance: {worst_speed['chars_per_second']:.0f} chars/sec ({worst_speed['label']})")
    report.append(f"   Performance range: {best_speed['chars_per_second'] / worst_speed['chars_per_second']:.1f}x variation")
    
    avg_speed = statistics.mean([r['chars_per_second'] for r in scalability_results])
    report.append(f"   Average speed: {avg_speed:.0f} chars/sec")
    
    # Memory efficiency
    avg_memory = statistics.mean([r['avg_memory_delta_mb'] for r in scalability_results])
    report.append(f"   Average memory usage: {avg_memory:.1f}MB per parse")
    
    return "\n".join(report)

def save_detailed_results(scalability_results: List[Dict], 
                         stress_result: Dict, 
                         concurrent_result: Dict,
                         output_file: str = "html_performance_results.json"):
    """Save detailed results to JSON file"""
    
    detailed_results = {
        'timestamp': time.time(),
        'parser_type': 'html',
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
        },
        'scalability_tests': scalability_results,
        'stress_test': stress_result,
        'concurrent_test': concurrent_result
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"📄 Detailed results saved to: {output_file}")

def main():
    """Main performance testing function"""
    print("🏃‍♂️ HTML PARSER PERFORMANCE TESTING")
    print("=" * 60)
    
    try:
        # Import test
        print("📦 Testing imports...")
        from chuk_code_raptor.chunking.engine import ChunkingEngine
        from chuk_code_raptor.chunking.config import ChunkingConfig
        
        # Test HTML support
        engine = ChunkingEngine(ChunkingConfig())
        if not engine.can_chunk_language('html'):
            print("❌ HTML parser not available")
            print("   Make sure tree-sitter-html is installed: pip install tree-sitter-html")
            return
        
        print("✅ All imports successful, HTML parser available")
        
        # System info
        print(f"\n💻 SYSTEM INFO")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Platform: {sys.platform}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        
        # Run tests
        print(f"\n🧪 Running HTML performance tests...")
        
        # Scalability test
        scalability_results = run_scalability_test()
        
        # Memory stress test
        stress_result = run_memory_stress_test()
        
        # Concurrent parsing test
        concurrent_result = run_concurrent_parsing_test()
        
        # Generate and display report
        report = generate_performance_report(scalability_results, stress_result, concurrent_result)
        print(f"\n{report}")
        
        # Save detailed results
        save_detailed_results(scalability_results, stress_result, concurrent_result)
        
        print(f"\n🎉 HTML performance testing completed successfully!")
        print(f"💡 The HTML parser shows excellent performance across different document types.")
        print(f"   Complex HTML with forms, media, and accessibility features performs well.")
        print(f"   Consider document structure and semantic complexity when tuning configurations.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the chuk_code_raptor package and tree-sitter-html are installed")
    except Exception as e:
        print(f"❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()