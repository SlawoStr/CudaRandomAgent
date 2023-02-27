#pragma once
#include "Agent.h"

class Wanderer : public Agent
{
public:
	__host__ __device__ Wanderer() : Agent{} {}

	__host__ __device__ Wanderer(float2 position, float angle) : Agent(position, angle) {}

	__host__ __device__ void wander(float maxSpeed, float maxForce, float wanderDistance, float wanderRadius, float randomAngle, float2 simulationBound)
	{
		float2 circlePos = m_velocity;
		float h = atan2(m_velocity.y, m_velocity.x);
		circlePos = normalize(circlePos);
		if (isnan(circlePos.x))
		{
			circlePos.x = 0.0;
			circlePos.y = 0.0;
		}
		circlePos *= wanderDistance;
		// Check if circle collide with simulation bound
		float2 wallBound = circlePos;
		wallBound += m_position;
		if (wallBound.x > simulationBound.x || wallBound.x < 0.f)
		{
			circlePos.x *= -1;
		}
		if (wallBound.y > simulationBound.y || wallBound.y < 0.f)
		{
			circlePos.y *= -1;
		}
		circlePos += m_position;
		circlePos.x += wanderDistance * cos(randomAngle + h);
		circlePos.y += wanderRadius * sin(randomAngle + h);
		seek(circlePos, maxSpeed, maxForce);
	}

private:
	__host__ __device__ void seek(float2 target, float maxSpeed, float maxForce)
	{
		target -= m_position;
		target = normalize(target);
		target *= maxSpeed;
		target -= m_velocity;
		target = limit(target, maxForce);
		applyForce(target);
	}
};