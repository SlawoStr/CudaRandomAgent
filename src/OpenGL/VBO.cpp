#include "VBO.h"

////////////////////////////////////////////////////////////
VBO::VBO(GLsizeiptr size)
{
	glGenBuffers(1, &ID);
	glBindBuffer(GL_ARRAY_BUFFER, ID);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
}

////////////////////////////////////////////////////////////
VBO::VBO(GLfloat* vertices, GLsizeiptr size)
{
	glGenBuffers(1, &ID);
	glBindBuffer(GL_ARRAY_BUFFER, ID);
	glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STREAM_DRAW);
}

void VBO::create(GLsizeiptr size)
{
	glGenBuffers(1, &ID);
	glBindBuffer(GL_ARRAY_BUFFER, ID);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
}

////////////////////////////////////////////////////////////
void VBO::Bind()
{
	glBindBuffer(GL_ARRAY_BUFFER, ID);
}

////////////////////////////////////////////////////////////
void VBO::Unbind()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

////////////////////////////////////////////////////////////
void VBO::Delete()
{
	glDeleteBuffers(1, &ID);
}

////////////////////////////////////////////////////////////
void VBO::reset(GLfloat* vertices, GLsizeiptr size)
{
	Bind();
	glBufferSubData(GL_ARRAY_BUFFER, 0, size, vertices);
	Unbind();
}