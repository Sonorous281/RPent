import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "RPent"
author = "RPent Contributors"
copyright = "2026, RPent Contributors"
version = "latest"
release = version

extensions = [
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
]

source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = []
default_role = "code"

language = "zh"
html_search_language = "zh"
html_theme = "pydata_sphinx_theme"
html_title = "RPent 中文文档"
html_show_sourcelink = False
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://rpent.readthedocs.io/zh-cn/latest/",
)
sitemap_url_scheme = "{link}"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = [
    "js/version-switcher.js",
    "js/lang-switcher.js",
    "js/sidebar-nav.js",
    "js/theme-toggle.js",
]
html_sidebars = {
    "**": [
        "sidebar-brand",
        "search-field",
        "sidebar-tools",
        "global-sidebar-nav",
    ]
}

html_theme_options = {
    "search_bar_text": "搜索文档…",
    "navbar_start": [],
    "navbar_center": [],
    "navbar_end": [],
    "navbar_align": "left",
    "secondary_sidebar_items": {"**": ["page-toc"], "index": []},
    "collapse_navigation": False,
    "show_nav_level": 1,
    "navigation_depth": 5,
    "header_links_before_dropdown": 10,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/RLinf/RPent",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ],
    "switcher": {
        "json_url": "_static/versions.json",
        "version_match": version,
    },
}


# ---------------------------------------------------------------------------
# 去除中文软换行渲染出的多余空格
#
# reStructuredText 会把源码里的换行渲染成一个空格。英文里没问题（单词本就
# 以空格分隔），但两个汉字被换行拆开时会多出一个空格。下面的钩子在 doctree
# 阶段删掉正文里相邻汉字之间的空格；代码块、行内 ``code``、raw HTML 以及
# 中英文之间的空格都不受影响。
# ---------------------------------------------------------------------------
import re as _re

from docutils import nodes as _nodes

_CJK = r"一-鿿㐀-䶿　-〿＀-￯"
_CJK_SPACE_RE = _re.compile(rf"([{_CJK}])\s+([{_CJK}])")


def _strip_cjk_spaces(app, doctree):
    skip = (
        _nodes.literal,
        _nodes.literal_block,
        _nodes.FixedTextElement,
        _nodes.raw,
        _nodes.comment,
        _nodes.math,
        _nodes.math_block,
    )
    for node in list(doctree.findall(_nodes.Text)):
        if isinstance(node.parent, skip):
            continue
        text = str(node)
        new = _CJK_SPACE_RE.sub(r"\1\2", text)
        new = _CJK_SPACE_RE.sub(r"\1\2", new)  # 再跑一次，处理连续重叠
        if new != text:
            node.parent.replace(node, _nodes.Text(new))


def setup(app):
    app.connect("doctree-read", _strip_cjk_spaces)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
