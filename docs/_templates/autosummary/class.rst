{% extends "!autosummary/class.rst" %}

{% block methods %} {% if methods %}
{% for item in methods %}
    .. automethod:: {{ item }}
{%- endfor %}
{% endif %} {% endblock %}

{% block attributes %} {% if attributes %}

{% endif %} {% endblock %}