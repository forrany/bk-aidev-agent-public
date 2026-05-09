# -*- coding: utf-8 -*-

from rest_framework import serializers


class AgentInfoSerializer(serializers.Serializer):
    agent_code = serializers.CharField(required=False, default=None)
