#pragma once
#include <glad/glad.h>

/// <summary>
/// VBO Class
/// </summary>
class VBO
{
public:
	VBO() {};

	/// <summary>
	/// Create VBO allocating memory for vertices uninitialized
	/// </summary>
	/// <param name="size">Size of memory for vertices allocation</param>
	VBO(GLsizeiptr size);

	/// <summary>
	/// Create VBO allocating memory for vertices initalized
	/// </summary>
	/// <param name="vertices">Value for allocated vertices</param>
	/// <param name="size">Size of memory for vertices allocation</param>
	VBO(GLfloat* vertices, GLsizeiptr size);

	/// <summary>
	/// Create VBO (constructor alternative)
	/// </summary>
	/// <param name="size"></param>
	void create(GLsizeiptr size);

	/// <summary>
	/// Bind VBO
	/// </summary>
	void Bind();

	/// <summary>
	/// Unbind VBO
	/// </summary>
	void Unbind();

	/// <summary>
	/// Delete VBO
	/// </summary>
	void Delete();

	/// <summary>
	/// Copy data to buffer
	/// </summary>
	/// <param name="vertices">New vertices values</param>
	/// <param name="size">Size of new vertices to copy</param>
	void reset(GLfloat* vertices, GLsizeiptr size);
public:
	GLuint ID;			//!< VBO ID
};