# Website / blog

### Portfolio Specifics
You can easily add full pages for each of the projects in your portfolio. If you want one to link to an external website, create a file for it in _portfolio, and  fil in the YAML front matter as you would for another, but with a redirect, like so:

	---
	layout: post
	title: Project
	description: a project that redirects to another website
	img:
	redirect: https://otherpage.com
	---
