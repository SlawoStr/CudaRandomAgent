#pragma once
#include "cuda_runtime.h"
#include "Utilities/VectorMathHelper.h"

class Agent
{
public:
	__host__ __device__ Agent() : m_position{}, m_velocity{}, m_acceleration{}, m_angle{}
	{}

	__host__ __device__ Agent(float2 position, float angle) : m_position(position), m_velocity{}, m_acceleration{}, m_angle{ angle }
	{}

	__host__ __device__ void update(float maxSpeed)
	{
		m_velocity += m_acceleration;
		m_velocity = limit(m_velocity, maxSpeed);
		m_acceleration = float2{ 0.0f,0.0f };
		m_position += m_velocity;
		m_angle = atan2(m_velocity.y, m_velocity.x);
	}

	__host__ __device__ void applyForce(float2 force)
	{
		m_acceleration += force;
	}

	__host__ __device__ float getAngle()const { return m_angle; }

	__host__ __device__ float2 getPosition() const { return m_position; }

protected:
	float2 m_position;
	float2 m_velocity;
	float2 m_acceleration;
	float  m_angle;
};